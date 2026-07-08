first run login.py -> create state.json 
state.json and scrapping code must be in the same folder.
output of scrapped excel, put it in the same folder of clustering file run as an input.

till now use single date tweet scrapping, becuase it will take too much time to 1 week or 1 month, 
so do repeat the single single day logic for 7 times to get 1 week, or 30 time to get 1 month.

also if the input excel is large then clustering and finding narrative code takes time.

for solve these 2 problems, if you want to fast the srapping, and processing , then there were 2-3 options:
***do not change the model for optimization or faster processing.
1. parallelize both code, means daywise scrapp will happen in different threads parallely.
   and for clustering, if possible after the cluster made summary and other things can be done parallely.
2. parallel different part of pipeline of narrative_discovery.py codefile.
3. use better pc.
4. change the order of work the pipeline does, but make sure logic and output don't change.
