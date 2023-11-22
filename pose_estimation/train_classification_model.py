import json
import scipy

if __name__ == "__main__":
    f = open("training_data.json")
    data = json.load(f)
    for i, obj in enumerate(data):
        print('Object', i, 'attacker', obj['attacker'][:10], 'gk', obj['goalkeeper'][:10])

    f.close()