from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings


# Sentences we want sentence embeddings for
files = pd.read_csv("file.csv")
trans_files = files.set_index(keys = "Questions").T
data_dict = trans_files.to_dict("list")


docs = []
for key in data_dict.keys():
	docs.append(key)

#Encode query and docs
doc_emb = encode(docs)


#while True:
#	query = input("Enter the query (type bye to exit) ->\n")
#	if query == "bye":
#		break
		

#	else:

#		query_emb = encode(query)

#		#Compute dot score between query and all document embeddings
#		scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

		#Combine docs & scores
#		doc_score_pairs = list(zip(docs, scores))

		#Sort by decreasing score
#		doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
		#print(doc_score_pairs[0][0])
#		answer = data_dict[doc_score_pairs[0][0]]
#		print(answer[0])
#		print()
#		print("#################################")
#		print("related questions ->")
#		print("1)" + doc_score_pairs[1][0])	
#		print("2)" + doc_score_pairs[2][0])
#		print("#################################")
#		print()
		#if query == "1":
		#	answer = data_dict[doc_score_pairs[1][0]]
		#	print(answer[0])
		#if query == "2":
		#	answer = data_dict[doc_score_pairs[2][0]]
		#	print(answer[0])

			#Output passages & scores
		#for doc, score in doc_score_pairs:
		#app = Flask(__name__, template_folder='template')


from flask import Flask, render_template, request
app = Flask(__name__, template_folder='template')

##define app routes
@app.route("/")
def index():
    	return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
	#doc_emb = initiate()
	#query = request.args.get('msg')
	#if query != "":
	#	query_emb = encode(query)
	#	answer = computeScores(doc_emb,query_emb)
	#	return str(answeri)
	query = request.args.get('msg')
	query_emb = encode(query)

	#Compute dot score between query and all document embeddings
	scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

	#Combine docs & scores
	doc_score_pairs = list(zip(docs, scores))

	#Sort by decreasing score
	doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
	if doc_score_pairs[0][1] > 0.4:
			print(doc_score_pairs[0])
			answer = data_dict[doc_score_pairs[0][0]]
			related = "<p>related questions -></p><p>1)"+ doc_score_pairs[1][0] +"</p><p>2)"+ doc_score_pairs[2][0] +"</p>"
			return "<b>BOT - </b>"+answer[0]+related



	else :
			answer = ["<p><b>BOT - </b>No suitable answer available :( </p><p>Try asking another meaningful question </p><p>Please ask covid-19 related questions only :)"]
			return answer[0]
	
if __name__ == "__main__":
	app.run()

