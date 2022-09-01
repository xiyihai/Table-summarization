# Table-summarization
Source code.
The source code of our domain-aware summarization algorithm is available in three python files, i.e., Summarization.py, NotabilityEva.py and functions.py. 
Notability.py is used to output the notability score for each entity.
Summarization.py is used to output a table summary.
function.py is used to provide the functions of obtaining the information needed in NotabilityEva.py including mapping entities to Wikipages, obtaining entity linkages between Wikipages, and retrieving the page views of entities.
The details of how to run these files are be found in each file.

Dataset.
Here is a collection of 993 real-world web tables crawled from popular websites:
https://downey-n1.cs.northwestern.edu/tableSearch/, 
www.imdb.com, 
www.thefamouspeople.com, 
www.nndb.com,
www.vgchartz.com.
These tables are mainly from 10 topics: Book, Dancer, Film, Music, Sports, Company, Food, Game, People, Software.
Note that for the simplity, we remove the attribute values of each tuple and only keep the primary entity column in these tables.
