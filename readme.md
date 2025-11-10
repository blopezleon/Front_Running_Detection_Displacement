<img width="2056" height="1329" alt="Image" src="https://github.com/user-attachments/assets/6ae6dffe-3de5-4f1f-8096-8c52181e8d27" />

# this project was hard, and vibe coding it made me want to off myself.
# examples:

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9e678180-a0cc-4eb7-82d6-bbae32f5bfee" width="450"/></td>
    <td><img src="https://github.com/user-attachments/assets/b800d59d-457c-4a5b-a9cc-a29e9406b216" width="450"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e1ed2a4e-45c2-4b86-b733-7c2932695fa2" width="450"/></td>
    <td><img src="https://github.com/user-attachments/assets/a1b2ade6-ee8f-482d-9b2a-1ba7a1338d6b" width="450"/></td>
  </tr>
</table>



# steps to get ts working:


1) create a virtual environment (preferably with conda, but `python -m venv .venv` will be fine) && run `pip install -r requirements.txt`.
2) run `python download_mempool_historical.py` to download most recent 7 days of mempool dumps. this will be like 40+ gigs of data (like 8-9 million txs).
3) run `python train_mev_detector_improved.py`. this will train a lightGBM model off of data labeled by the flashboys algorithm.
4) run `python analyze_auction_timing.py` to see how early detections can be made. don't get too excited.
5) run `python live_mempool_scorer.py`. when it doesn't that well, don't ask me cuz i don't fucking know how to fix it either.
