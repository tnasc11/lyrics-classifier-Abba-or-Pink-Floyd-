import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--write', help='song line to predict')
args = parser.parse_args()

# get the pickled model (trained model)
with open('pipeline.pkl', 'rb') as pickle_file:
	pipeline = pickle.load(pickle_file)

# get the lyrics from the terminal (option `-w`)
if args.write:
	prediction = pipeline.predict([args.write])
	probability = pipeline.predict_proba([args.write])
	print(prediction)

print(f'the artist is {prediction} with {probability} probability')