from src.train.data import load_raw_data

if __name__ == "__main__":
    data = load_raw_data()
    data.sample(n=20).to_json('sample.json')
