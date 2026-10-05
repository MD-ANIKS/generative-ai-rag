import arxiv 

client = arxiv.Client()

# create the retriever 
search = arxiv.Search(
  max_results = 2, # number of papers to retrieve
  query="large language models"
)


#print results 
for result in client.results(search):
  print("Title:", result.title)
  print("Authors:", result.authors)
  print("Summary:", result.summary[:500]) #print first 500 char