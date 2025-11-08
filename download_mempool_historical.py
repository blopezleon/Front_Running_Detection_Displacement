#!/usr/bin/env python3

import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "https://mempool-dumpster.flashbots.net/ethereum/mainnet"
DATA_DIR = Path("data/mempool")

def download_file(url, filepath):
    if filepath.exists():
        logger.info(f"File exists: {filepath.name}")
        return True
    
    try:
        logger.info(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def get_date_range(days=7):
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates

def download_mempool_data(dates):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    failed = []
    
    for date in dates:
        year_month = date[:7]
        
        parquet_url = f"{BASE_URL}/{year_month}/{date}.parquet"
        parquet_path = DATA_DIR / f"{date}.parquet"
        
        if download_file(parquet_url, parquet_path):
            downloaded.append(parquet_path)
        else:
            failed.append(date)
    
    logger.info(f"\nDownload complete:")
    logger.info(f"  Success: {len(downloaded)} files")
    logger.info(f"  Failed: {len(failed)} files")
    
    if downloaded:
        total_size = sum(f.stat().st_size for f in downloaded) / 1e9
        logger.info(f"  Total size: {total_size:.2f} GB")
    
    return downloaded, failed

def main():
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    else:
        days = 7
    
    logger.info(f"Downloading {days} days of mempool data from Flashbots")
    logger.info(f"Data directory: {DATA_DIR.absolute()}")
    
    dates = get_date_range(days)
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    downloaded, failed = download_mempool_data(dates)
    
    if failed:
        logger.warning(f"Failed dates: {', '.join(failed)}")
    
    logger.info("\nTo download more data, run:")
    logger.info(f"  python {sys.argv[0]} <number_of_days>")

if __name__ == "__main__":
    main()
