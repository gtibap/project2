import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

######
# read
def read_in_data():
##
	#read data
	train_X=pd.DataFrame.from_csv('data/forExample.csv')
	train_y= pd.DataFrame.from_csv('data/forExample_labels.csv')
##	
	# data + categories (class)
	df=pd.concat([train_X, train_y], axis=1)
	df.columns = ['Text', 'category']
##	
	#clean data and become lowercase
	df = df[df.Text.str.contains("http")== False  ] # removing lines that contain http
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df['Text'] = df['Text'].str.lower()
	df=df.fillna(method='ffill')
	df = df[df['Text'].str.rstrip()!=''] # dropping empty strings, df
	df.dropna(axis=0, how='any',inplace=True) 
##	
# copy labels
	r = df['category'].tolist()
	#r = df[['category']].copy()
	#r.category = r.category.astype(int)
	#print "r: \n", r
	#split data to training and validation
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=42)

	# extract features from text (count of characters)
	#text_list = df['Text'].tolist()
	text_train = X_train['Text'].tolist()
	text_test = X_test['Text'].tolist()
	
	#print "data_X: ", text_list
	# count characters TRAIN
	vectorizer = CountVectorizer(analyzer='char')
	#vectorizer_train = CountVectorizer(analyzer='char')
	#vectorizer_test = CountVectorizer(analyzer='char')
	
	vect_train = vectorizer.fit_transform(text_train)
	vocab = vectorizer.get_feature_names()
	print "voc: \n", vocab
	
	# count characters TEST with features from TRAIN (vocabulry)
	vectorizer = CountVectorizer(analyzer='char', vocabulary=vocab)
	vect_test  = vectorizer.fit_transform(text_test)
	
	X_train = vect_train.toarray()
	X_test = vect_test.toarray()
##	
	
	
	print X_train
	print X_train.shape
	print y_train
	
	print X_test
	print X_test.shape
	print y_test
	
	
	
	return X_train

####
# MAIN

print "Start ..."
text_list = read_in_data()



	
	

