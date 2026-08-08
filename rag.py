import os
import pickle
import json
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

# =====================================================
# Configuration
# =====================================================

CSV_FILE = "amazon_products.csv"
INDEX_FILE = "faiss_index.index"
DATA_FILE = "products.pkl"
CACHE_FILE = "category_cache.json"
IMAGE_INDEX_FILE = "image_index.index"
IMAGE_EMBEDDINGS_FILE = "image_embeddings.pkl"

NUMBER_OF_PRODUCTS = 5000

# =====================================================
# Category Translation (unchanged)
# =====================================================

def load_category_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_category_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def translate_category(category, cache):
    if category in cache:
        return cache[category]
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(category)
        cache[category] = translated
        return translated
    except Exception as e:
        print(f"Translation error for '{category}': {e}")
        cache[category] = category
        return category

def get_all_categories(products, auto_translate=True):
    raw_categories = sorted(products["categoryName"].unique())
    if not auto_translate:
        return raw_categories
    cache = load_category_cache()
    translated = {}
    for cat in raw_categories:
        translated[cat] = translate_category(cat, cache)
    save_category_cache(cache)
    return sorted(set(translated.values()))

def get_category_mapping(products):
    raw_categories = sorted(products["categoryName"].unique())
    cache = load_category_cache()
    mapping = {}
    for cat in raw_categories:
        mapping[cat] = translate_category(cat, cache)
    save_category_cache(cache)
    return mapping

def get_reverse_category_mapping(products):
    mapping = get_category_mapping(products)
    reverse = {}
    for hindi, eng in mapping.items():
        reverse[eng] = hindi
    return reverse

# =====================================================
# Title Translation – FIXED (no cache)
# =====================================================

def translate_text(text, dest='en'):
    """
    Translate text using a fresh GoogleTranslator instance.
    Returns the translated text, or the original if translation fails.
    """
    if not text or not text.strip():
        return text
    try:
        translator = GoogleTranslator(source='auto', target=dest)
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# =====================================================
# Load model and database (unchanged)
# =====================================================

def load_model():
    print("Loading Sentence Transformer model...")
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_database():
    if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
        print("Loading existing FAISS index...")
        index = faiss.read_index(INDEX_FILE)
        with open(DATA_FILE, "rb") as f:
            df = pickle.load(f)
        return index, df

    print("Creating new FAISS index...")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"❌ {CSV_FILE} not found.")

    df = pd.read_csv(CSV_FILE)
    sample_size = min(NUMBER_OF_PRODUCTS, len(df))
    df = df.sample(n=sample_size, random_state=42)

    columns = [
        "asin",
        "title",
        "categoryName",
        "price",
        "listPrice",
        "stars",
        "reviews",
        "isBestSeller",
        "imgUrl",
        "productURL"
    ]
    df = df[[c for c in columns if c in df.columns]]
    df.fillna("", inplace=True)

    df["document"] = (
        "Title: " + df["title"].astype(str) +
        " Category: " + df["categoryName"].astype(str) +
        " Price: ₹" + df["price"].astype(str) +
        " Original Price: ₹" + df["listPrice"].astype(str) +
        " Rating: " + df["stars"].astype(str) +
        " Reviews: " + df["reviews"].astype(str) +
        " Best Seller: " + df["isBestSeller"].astype(str)
    )

    model = load_model()
    print("Generating text embeddings...")
    embeddings = model.encode(
        df["document"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )
    embeddings = embeddings.astype("float32")

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("Saving FAISS index and product data...")
    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(df, f)

    print(f"✅ Database created with {len(df)} products.")
    return index, df

def search_products(query, model, index, products, top_k=20, translate_titles=False):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    cat_mapping = get_category_mapping(products)
    for idx in indices[0]:
        row = products.iloc[idx]
        try:
            price = float(row["price"])
        except:
            price = 0
        try:
            list_price = float(row["listPrice"])
        except:
            list_price = 0
        if list_price > 0:
            discount = round(((list_price - price) / list_price) * 100)
        else:
            discount = 0

        category = cat_mapping.get(row["categoryName"], row["categoryName"])
        title = row["title"]
        if translate_titles:
            title = translate_text(title, dest='en')

        results.append({
            "asin": row["asin"],
            "title": title,
            "category": category,
            "price": price,
            "listPrice": list_price,
            "discount": discount,
            "stars": row["stars"],
            "reviews": row["reviews"],
            "bestSeller": row["isBestSeller"],
            "image": row["imgUrl"],
            "url": row["productURL"]
        })
    return results

def get_price_range(products):
    prices = pd.to_numeric(products["price"], errors='coerce')
    return float(prices.min()), float(prices.max())

def get_products_by_category(products, category_english, top_k=5):
    reverse_mapping = get_reverse_category_mapping(products)
    original_cat = reverse_mapping.get(category_english)
    if original_cat is None:
        for eng, hin in reverse_mapping.items():
            if category_english.lower() in eng.lower() or eng.lower() in category_english.lower():
                original_cat = hin
                break
    if original_cat is None:
        print(f"Category '{category_english}' not found in mapping.")
        return []

    filtered = products[products["categoryName"] == original_cat]
    if len(filtered) == 0:
        return []
    sample = filtered.head(top_k)
    results = []
    cat_mapping = get_category_mapping(products)
    for idx, row in sample.iterrows():
        try:
            price = float(row["price"])
        except:
            price = 0
        try:
            list_price = float(row["listPrice"])
        except:
            list_price = 0
        if list_price > 0:
            discount = round(((list_price - price) / list_price) * 100)
        else:
            discount = 0
        category_en = cat_mapping.get(row["categoryName"], row["categoryName"])
        results.append({
            "asin": row["asin"],
            "title": row["title"],
            "category": category_en,
            "price": price,
            "listPrice": list_price,
            "discount": discount,
            "stars": row["stars"],
            "reviews": row["reviews"],
            "bestSeller": row["isBestSeller"],
            "image": row["imgUrl"],
            "url": row["productURL"]
        })
    return results

# =====================================================
# Image Search (CLIP) – updated with translation support
# =====================================================

_clip_model = None
_clip_processor = None

def load_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            print("Loading CLIP model (this may take a moment)...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
        except Exception as e:
            print(f"ERROR loading CLIP: {e}")
            _clip_model = None
            _clip_processor = None
    return _clip_model, _clip_processor

def get_image_embedding(image_url_or_pil):
    model, processor = load_clip_model()
    if processor is None:
        raise RuntimeError("CLIP processor failed to load. Check your internet connection and try again.")
    
    if isinstance(image_url_or_pil, str):
        try:
            response = requests.get(image_url_or_pil, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Failed to fetch image: {e}")
            return None
    else:
        img = image_url_or_pil.convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.squeeze().numpy().astype("float32")

def build_image_index(products):
    if os.path.exists(IMAGE_INDEX_FILE) and os.path.exists(IMAGE_EMBEDDINGS_FILE):
        print("Loading existing image index...")
        image_index = faiss.read_index(IMAGE_INDEX_FILE)
        with open(IMAGE_EMBEDDINGS_FILE, "rb") as f:
            valid_indices = pickle.load(f)
        return image_index, valid_indices

    print("Building image index (this may take a while)...")
    valid_mask = products["imgUrl"].str.startswith("http", na=False)
    valid_products = products[valid_mask]
    if len(valid_products) == 0:
        print("No valid image URLs found. Image search disabled.")
        return None, []

    embeddings_list = []
    valid_indices = []
    load_clip_model()
    for i, (idx, row) in enumerate(valid_products.iterrows()):
        url = row["imgUrl"]
        try:
            emb = get_image_embedding(url)
            if emb is not None:
                embeddings_list.append(emb)
                valid_indices.append(i)
        except Exception as e:
            print(f"Error on image {i}: {e}")
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(valid_products)} images")

    if not embeddings_list:
        print("No image embeddings could be generated.")
        return None, []

    embeddings = np.array(embeddings_list).astype("float32")
    dimension = embeddings.shape[1]
    image_index = faiss.IndexFlatL2(dimension)
    image_index.add(embeddings)

    faiss.write_index(image_index, IMAGE_INDEX_FILE)
    with open(IMAGE_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(valid_indices, f)

    print(f"✅ Image index built with {len(embeddings)} images.")
    return image_index, valid_indices

def search_by_image(image, products, top_k=5, translate_titles=False):
    """
    Search similar products by image.
    If translate_titles=True, product titles are translated to English.
    """
    image_index, valid_indices = build_image_index(products)
    if image_index is None:
        return []

    emb = get_image_embedding(image)
    if emb is None:
        return []

    emb = emb.reshape(1, -1)
    distances, indices = image_index.search(emb, top_k)

    valid_mask = products["imgUrl"].str.startswith("http", na=False)
    valid_products = products[valid_mask]
    results = []
    cat_mapping = get_category_mapping(products)
    for idx in indices[0]:
        if idx < len(valid_indices):
            orig_idx = valid_indices[idx]
            row = valid_products.iloc[orig_idx]
            try:
                price = float(row["price"])
            except:
                price = 0
            try:
                list_price = float(row["listPrice"])
            except:
                list_price = 0
            if list_price > 0:
                discount = round(((list_price - price) / list_price) * 100)
            else:
                discount = 0

            category = cat_mapping.get(row["categoryName"], row["categoryName"])
            title = row["title"]
            if translate_titles:
                title = translate_text(title, dest='en')
            results.append({
                "asin": row["asin"],
                "title": title,
                "category": category,
                "price": price,
                "listPrice": list_price,
                "discount": discount,
                "stars": row["stars"],
                "reviews": row["reviews"],
                "bestSeller": row["isBestSeller"],
                "image": row["imgUrl"],
                "url": row["productURL"]
            })
    return results

# =====================================================
# Main (for testing)
# =====================================================

def main():
    print("\n" + "="*60)
    print("🛒 ShopWise AI – Index Builder & Test Mode")
    print("="*60 + "\n")

    model = load_model()
    index, products = load_database()

    print(f"\n✅ Ready! {len(products)} products indexed.\n")
    print("Commands: text query, 'image' to test image search, 'exit' to quit.\n")

    while True:
        q = input("🔍 Query: ")
        if q.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        if not q.strip():
            continue

        if q.lower() == "image":
            img_path = input("Enter path to image file: ")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                results = search_by_image(img, products, top_k=5, translate_titles=True)
                if results:
                    for i, p in enumerate(results, 1):
                        print(f"{i}. {p['title']} – ₹{p['price']}")
                else:
                    print("No similar products found.")
            else:
                print("File not found.")
            continue

        trans = input("Translate titles to English? (y/n): ").lower() == 'y'
        results = search_products(q, model, index, products, top_k=5, translate_titles=trans)

        print(f"\n📦 Top {len(results)} results:\n")
        for i, p in enumerate(results, 1):
            print(f"{i}. {p['title']}")
            print(f"   Price: ₹{p['price']} (was ₹{p['listPrice']}, {p['discount']}% off)")
            print(f"   Rating: {p['stars']} ⭐ | {p['reviews']} reviews")
            print(f"   Category: {p['category']}")
            print(f"   URL: {p['url']}")
            print("-" * 50)
        print()

if __name__ == "__main__":
    main()