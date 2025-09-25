# -*- coding: utf-8 -*-
"""
Import CSV -> MySQL base (tables: brand, cosmetics, retailer_offers)

รองรับไฟล์:
1) cosmetics_products.csv
   - คอลัมน์ขั้นต่ำ: brand, category, name, shade
   - ตัวเลือก: skin_tone, price, retailer_links (ลิงก์คั่นด้วย | หรือ , หรือขึ้นบรรทัดใหม่)

2) brand_shade_map.csv (official)
   - brand,category,product_name,shade_label,skin_tone_official,price_thb[,image_url,product_link]

3) overrides.csv (admin override)
   - brand,category,product_name,shade_label,skin_tone_override,price_override

4) retailer_links.csv (ลิงก์แยกไฟล์)
   - brand,category,product_name,shade_label,retailer,link1,link2,link3,is_official

ตาราง DB ที่คาดหวัง:
- brand(brandID, brandName)
- cosmetics(CosmeticID, BrandID, Name, Type, ShadeCode, ShadeName, Price, ImageURL, ProductLink, suitableSkinTone)
- retailer_offers(OfferID, CosmeticID, Retailer, URL, ImageURL, PriceTHB, Rating, ReviewCount, IsOfficial, LastUpdate)
  * Retailer อาจเป็น ENUM('shopee','lazada','sephora','watsons','other','legacy')

หมายเหตุ:
- ใช้ INSERT IGNORE กันลิงก์ซ้ำ (ควรมี unique (CosmeticID, URL(191)) หรือ hash ในฐาน)
- map retailer จากโดเมน URL -> enum ให้เอง; ไม่แมตช์จะตกไป 'other'
"""

import re
from pathlib import Path
from typing import Optional, List

import pandas as pd
import mysql.connector as mysql

# ------------------ DB CONFIG ------------------
DB = dict(
    host="127.0.0.1",
    user="root",
    password="1234",
    database="db_miniprojectfinal",
    autocommit=False,
)

# ------------------ FILE NAMES -----------------
FILE_PRODUCTS = "cosmetics_products.csv"
FILE_OFFICIAL = "brand_shade_map.csv"
FILE_OVERRIDE = "overrides.csv"
FILE_LINKS    = "retailer_links.csv"

# ------------------ CONSTANTS ------------------
# รองรับโค้ดเฉดของแบรนด์หลัก ๆ (MAC/NARS/Fenty ฯลฯ)
SHADE_CODE_PAT = re.compile(r'^(?:NC|NW|N|C|W)?\d+[A-Z]*$|^\d{2,3}[A-Z]*$')

# ENUM ของตาราง retailer_offers.Retailer (ถ้าตารางคุณเป็น ENUM ควรตรงชุดนี้)
RETAILER_ENUM = {'shopee','lazada','sephora','watsons','other','legacy'}

# เผื่อ URL เกินความยาวคอลัมน์ในฐาน (เช่น 1000 หรือ 1024)
MAX_URL_LEN = 1000  # ปรับเป็น 1024 หากคอลัมน์คุณยาว 1024


# ------------------ HELPERS --------------------
def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def is_code(shade_label: str) -> bool:
    s = norm(shade_label)
    return bool(s and SHADE_CODE_PAT.match(s))

def read_csv_any(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def clean_price(x) -> Optional[float]:
    if pd.isna(x) or x is None:
        return None
    s = str(x)
    s = s.replace(',', '')
    s = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
    try:
        return float(s) if s else None
    except Exception:
        return None

def split_links(val: str) -> List[str]:
    if pd.isna(val) or not str(val).strip():
        return []
    raw = str(val).replace('\n', '|').replace(',', '|')
    links = []
    for u in raw.split('|'):
        u = u.strip()
        if not u or u.lower() == 'nan':
            continue
        # truncate เพื่อกัน error key length
        links.append(u[:MAX_URL_LEN])
    return links

def norm_retailer_name_from_url(url: str) -> str:
    u = (url or '').lower()
    if 'shopee.' in u:  return 'shopee'
    if 'lazada.' in u:  return 'lazada'
    if 'sephora.' in u: return 'sephora'
    if 'watsons' in u:  return 'watsons'
    return 'other'

def norm_retailer(name: str) -> str:
    s = (name or '').strip().lower()
    mapping = {
        'shoppee':'shopee', 'shp':'shopee',
        'lz':'lazada',
        'watson':'watsons',
        'official':'other', 'brand':'other', 'website':'other'
    }
    s = mapping.get(s, s)
    return s if s in RETAILER_ENUM else 'other'


# ------------------ DB OPS ---------------------
def upsert_brand(cur, brand_name: str) -> int:
    cur.execute("INSERT IGNORE INTO brand(brandName) VALUES(%s)", (brand_name,))
    cur.execute("SELECT brandID FROM brand WHERE brandName=%s", (brand_name,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"brand not found/created: {brand_name}")
    return row[0]

def get_or_create_cosmetic(cur,
                           brand_id: int,
                           name: str, ctype: str,
                           shade_label: str,
                           price: Optional[float] = None,
                           image_url: Optional[str] = None,
                           product_link: Optional[str] = None,
                           tone: Optional[str] = None,
                           description: Optional[str] = None) -> int:
    shade_code = shade_label if is_code(shade_label) else None
    shade_name = None if shade_code else shade_label

    # หา record เดิม
    cur.execute("""
      SELECT CosmeticID FROM cosmetics
      WHERE BrandID=%s AND Name=%s AND Type=%s
        AND COALESCE(ShadeCode,'') = COALESCE(%s,'')
        AND COALESCE(ShadeName,'') = COALESCE(%s,'')
    """, (brand_id, name, ctype, shade_code, shade_name))
    row = cur.fetchone()
    if row:
        cid = row[0]
        sets, params = [], []
        if price is not None:
            sets += ["Price=%s"]; params += [price]
        if image_url:
            sets += ["ImageURL=%s"]; params += [image_url]
        if product_link:
            sets += ["ProductLink=%s"]; params += [product_link]
        if tone:
            sets += ["suitableSkinTone=%s"]; params += [tone]
        if description:
            sets += ["Description=%s"]; params += [description]
        if sets:
            sql = f"UPDATE cosmetics SET {', '.join(sets)} WHERE CosmeticID=%s"
            params += [cid]
            cur.execute(sql, params)
        return cid

    # INSERT ใหม่ (ใส่ Description ด้วย)
    cur.execute("""
      INSERT INTO cosmetics
        (BrandID, Name, Type, ShadeCode, ShadeName, Price, ImageURL, ProductLink, suitableSkinTone, Description)
      VALUES
        (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (brand_id, name, ctype, shade_code, shade_name, price, image_url, product_link, tone, description))
    return cur.lastrowid


def add_offer(cur, cosmetic_id: int, retailer: str, url: str,
              is_official: int = 0,
              price_thb: Optional[float] = None,
              image_url: Optional[str] = None,
              rating: Optional[float] = None,
              review_count: Optional[int] = None):
    if not url:
        return
    # truncate เผื่อคอลัมน์สั้นกว่า MAX_URL_LEN
    url = url[:MAX_URL_LEN]
    # กันลิงก์ซ้ำต่อสินค้า (ใช้ INSERT IGNORE)
    cur.execute("""
      INSERT IGNORE INTO retailer_offers
      (CosmeticID, Retailer, URL, IsOfficial, PriceTHB, ImageURL, Rating, ReviewCount)
      VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
    """, (cosmetic_id, retailer or 'other', url, int(is_official or 0),
          price_thb, image_url, rating, review_count))


# ------------------ PIPELINES ------------------
def import_from_products_csv(cur, path=FILE_PRODUCTS):
    p = Path(path)
    if not p.exists():
        print(f"⚠ ไม่พบไฟล์ {path} (ข้าม)")
        return

    df = pd.read_csv(p, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    required = {'brand','category','name','shade'}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} ต้องมีคอลัมน์อย่างน้อย: {sorted(required)}")

    has_links = 'retailer_links' in df.columns
    upsert_products, inserted_offers, skipped_offers = 0, 0, 0

    for _, r in df.iterrows():
        brand = norm(r.get('brand',''))
        ctype = norm(r.get('category',''))
        pname = norm(r.get('name',''))
        shade = norm(r.get('shade',''))
        tone  = norm(r.get('skin_tone','')) or None
        price = clean_price(r.get('price', None))
        desc  = norm(r.get('description','')) or None

        bid = upsert_brand(cur, brand)
        cid = get_or_create_cosmetic(cur, bid, pname, ctype, shade,
                                     price=price, image_url=None, product_link=None, tone=tone, description=desc)
        upsert_products += 1

        if has_links:
            for url in split_links(r['retailer_links']):
                retailer = norm_retailer_name_from_url(url)
                if retailer not in RETAILER_ENUM:
                    retailer = 'other'
                try:
                    add_offer(cur, cid, retailer, url, is_official=0, price_thb=None)
                    if cur.rowcount:
                        inserted_offers += 1
                    else:
                        skipped_offers += 1  # duplicate (unique ชน)
                except mysql.Error as e:
                    skipped_offers += 1
                    print(f"[offer-skip] {e} | {url[:120]}")

    print(f"✔ products.csv: upsert_products={upsert_products}, offers_inserted={inserted_offers}, offers_skipped={skipped_offers}")


def import_official(cur, df_off: pd.DataFrame):
    need = {'brand','category','product_name','shade_label','skin_tone_official'}
    if not need.issubset(df_off.columns):
        raise ValueError(f"brand_shade_map.csv missing columns: {sorted(need - set(df_off.columns))}")

    added, updated = 0, 0
    for _, r in df_off.iterrows():
        brand = norm(r['brand'])
        ctype = norm(r['category'])
        pname = norm(r['product_name'])
        shade = norm(r['shade_label'])
        tone  = norm(r['skin_tone_official']) or None
        price = clean_price(r.get('price_thb', None))
        img   = norm(r.get('image_url', '')) or None
        plink = norm(r.get('product_link', '')) or None
        desc  = norm(r.get('description','')) or None

        bid = upsert_brand(cur, brand)
        before_count = cur.rowcount
        cid = get_or_create_cosmetic(cur, bid, pname, ctype, shade,
                                     price=price, image_url=img, product_link=plink, tone=tone, description=desc)
        # ประมาณการ insert/update
        if cur.rowcount and cur.lastrowid == cid and before_count != cur.rowcount:
            added += 1
        else:
            updated += 1
    print(f"✔ official: ~added={added}, ~updated={updated}")


def apply_overrides(cur, df_ovr: pd.DataFrame):
    need = {'brand','category','product_name','shade_label'}
    if not need.issubset(df_ovr.columns):
        raise ValueError(f"overrides.csv missing columns: {sorted(need - set(df_ovr.columns))}")

    changed = 0
    for _, r in df_ovr.iterrows():
        brand = norm(r['brand']); ctype = norm(r['category'])
        pname = norm(r['product_name']); shade = norm(r['shade_label'])
        tone_o = norm(r.get('skin_tone_override', ''))
        price_o = clean_price(r.get('price_override', None))

        bid = upsert_brand(cur, brand)
        cid = get_or_create_cosmetic(cur, bid, pname, ctype, shade)  # ensure exists

        sets, params = [], []
        if tone_o:
            sets += ["suitableSkinTone=%s"]; params += [tone_o]
        if price_o is not None:
            sets += ["Price=%s"]; params += [price_o]
        if sets:
            sql = f"UPDATE cosmetics SET {', '.join(sets)} WHERE CosmeticID=%s"
            params += [cid]
            cur.execute(sql, params)
            changed += cur.rowcount
    print(f"✔ overrides: changed_rows={changed}")


def import_links(cur, df_links: pd.DataFrame):
    need = {'brand','category','product_name','shade_label'}
    if not need.issubset(df_links.columns):
        raise ValueError(f"retailer_links.csv missing columns: {sorted(need - set(df_links.columns))}")

    inserted, skipped = 0, 0
    for _, r in df_links.iterrows():
        brand = norm(r['brand']); ctype = norm(r['category'])
        pname = norm(r['product_name']); shade = norm(r['shade_label'])
        retailer = norm_retailer(r.get('retailer', ''))
        is_off   = int(r['is_official']) if 'is_official' in r and pd.notna(r['is_official']) else 0

        bid = upsert_brand(cur, brand)
        cid = get_or_create_cosmetic(cur, bid, pname, ctype, shade)

        for k in ('link1','link2','link3'):
            url = norm(r.get(k, ''))
            if not url:
                continue
            url = url[:MAX_URL_LEN]
            try:
                add_offer(cur, cid, retailer, url, is_official=is_off)
                if cur.rowcount:
                    inserted += 1
                else:
                    skipped += 1  # duplicate
            except mysql.Error as e:
                skipped += 1
                print(f"[offer-skip] {e} | {url[:120]}")
    print(f"✔ links.csv: inserted={inserted}, skipped={skipped}")


# ------------------ MAIN -----------------------
def main():
    conn = mysql.connect(**DB)
    cur  = conn.cursor()

    try:
        # 1) นำเข้าจากไฟล์รวม (เร็วและพอใช้งานได้ทันที)
        import_from_products_csv(cur, FILE_PRODUCTS)

        # 2) (ตัวเลือก) official + overrides + links (ถ้ามี จะทับ/เติมข้อมูล)
        df_off = read_csv_any(FILE_OFFICIAL)
        if df_off is not None:
            import_official(cur, df_off)

        df_ovr = read_csv_any(FILE_OVERRIDE)
        if df_ovr is not None:
            apply_overrides(cur, df_ovr)

        df_lnk = read_csv_any(FILE_LINKS)
        if df_lnk is not None:
            import_links(cur, df_lnk)

        conn.commit()
        print("✅ DONE: ข้อมูลถูกบันทึกลงฐานเรียบร้อย")
    except Exception as e:
        conn.rollback()
        print("❌ ROLLBACK:", e)
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
