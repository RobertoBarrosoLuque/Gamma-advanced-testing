from src.train.data import load_raw_data, DATA_DIR

if __name__ == "__main__":
    data = load_raw_data()
    data.sample(n=20).to_json(DATA_DIR / 'sample.json')
