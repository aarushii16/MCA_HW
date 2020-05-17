import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
	"""
	relevance feedback
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		n: integer
			number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""
	alpha=0.75
	beta=0.25

	gt=[[] for i in range(30)]
	f=open("data/MED.REL","r")
	for t in range(696):
		try:
			x=f.readline()
			x=x.split(" ")
			i = int(x[0])
			gt[i-1].append(int(x[2]))
		except:
			pass

	copy_vecqueries=vec_queries

	for epoch in range(3):
		for i in range(sim.shape[1]):
			# n=len(gt[i])
			rel=np.argsort(-sim[:, i])[:n]
			rs=set(rel)
			# print(rs)
			gs=set(gt[i])
			# print(gs)
			relevant=rs.intersection(gs)
			# print(relevant)
			nonrelevant = rs-relevant
			# print(nonrelevant)

			alphasum=0
			betasum=0

			for t in relevant:
				alphasum=alphasum+vec_docs[t-1][:]
			# print(alphasum.shape)

			for t in nonrelevant:
				betasum=betasum+vec_docs[t-1][:]
			# print(betasum.shape)

			copy_vecqueries[i]=vec_queries[i]+ (alpha*alphasum) - (beta*betasum)

		vec_queries=copy_vecqueries

	rf_sim = cosine_similarity(vec_docs, vec_queries)

	# print(vec_docs.shape)
	# print(sim.shape)
	# print(vec_queries.shape)
	return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
	"""
	relevance feedback with expanded queries
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		tfidf_model: TfidfVectorizer,
			tf_idf pretrained model
		n: integer
			number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""

	alpha=0.75
	beta=0.25

	gt=[[] for i in range(30)]
	f=open("data/MED.REL","r")
	for t in range(696):
		try:
			x=f.readline()
			x=x.split(" ")
			i = int(x[0])
			gt[i-1].append(int(x[2]))
		except:
			pass

	copy_vecqueries=vec_queries

	for epoch in range(3):
		for i in range(sim.shape[1]):
			# n=len(gt[i])
			rel=np.argsort(-sim[:, i])[:n]
			rs=set(rel)
			# print(rs)
			gs=set(gt[i])
			# print(gs)
			relevant=rs.intersection(gs)
			# print(relevant)
			nonrelevant = rs-relevant
			# print(nonrelevant)

			alphasum=0
			betasum=0

			for t in relevant:
				v=(vec_docs[t-1]).toarray()
				d={}
				for ii in range(v.shape[1]):
					if v[0][ii]!=0:
						d[ii]=v[0][ii]
				d=sorted(d.items(),key=lambda kv: (kv[1], kv[0]))
				to_add=[0 for tt in range(10625)]
				ind=0
				for ll in d:
					to_add[ll[0]]=ll[1]
					ind=ind+1
					if ind==10:
						break
					# print(to_add[ll[0]],ll[1])
				# print(to_add)
				alphasum=alphasum+np.asarray(to_add)
			# print(alphasum.shape)

			for t in nonrelevant:
				v=(vec_docs[t-1]).toarray()
				d={}
				for ii in range(v.shape[1]):
					if v[0][ii]!=0:
						d[ii]=v[0][ii]
				d=sorted(d.items(),key=lambda kv: (kv[1], kv[0]))
				to_add=[0 for tt in range(10625)]
				ind=0
				for ll in d:
					to_add[ll[0]]=ll[1]
					ind=ind+1
					if ind==10:
						break
				# print(to_add.shape)
				betasum=betasum+np.asarray(to_add)
			# print(betasum.shape)

			copy_vecqueries[i]=vec_queries[i]+ (alpha*alphasum) - (beta*betasum)
			

		vec_queries=copy_vecqueries

	rf_sim = cosine_similarity(vec_docs, vec_queries)

	return rf_sim