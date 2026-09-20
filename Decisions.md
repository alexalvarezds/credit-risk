# DECISIONS TAKEN
Here are the Decisions taken on the project, with the objective to have it all documented and no decision is made out ouf nowhere or unexplained

# About Leakage
- The final dataset consists of observations of unique clients, so making aggregates of them wont cause leakage
- But using info of all clients to compute numbers is leakage because i would be using test clients info
- So on the first step until i have all of it cleaned, just do aggregate variables and put it all so i can join it and later do the partition and start the funny part

# Methodology
- First clean them all and make aggregate metrics resuming the tables, so i can join them
- 

# application_train.csv
- Treat ir first so it is the main table, do basic stuff to clean data, quit all the NA, basic feature encoding, basic stuff
- Leave Kaprimont to see if different categories do affect the target or they are all basically the same, and later check how they correlate with the target

# installments_payments.csv
- First explore it, because it is historical data about other loans and not the actual loan there is no leakage as we are not using information about the state of the actual loan, and all the information is not mixed on the temporal dimension with the actual loans, but first check it
- I could make variables with this about relative fully repay date, if someone payed on date, before or after, if they have a loan active right now, percentage of total amount repayed, average days took to repay it fully, maximum days it took to repay it across all the loans and a metric determining if on each loan they're behavior is getting better or worse, this is they pay sooner or later or they dont pay at all
- I could extract out of total loans how many have they fully repayed, maybe a proportion of loans_repayed/total_loans or money_repayed/total_money_topay
- Group by ids then get all of the aggregate variables, if they have another active loan, rate of impaid quotes
- On PAYMENT_DIFF instead of leaving the NaN, putting the whole cuota unpaid
- Using the ratio between the actual quantity and the expected quantity so later the model can see the number and the proportion

# previous_application.csv
- We can't impute some value on the rate ones, it will create a high bias, on other words we would be practically creating it from scratch
- For the EDA, look on DOWN_AMT_PAYMENT and RATE_AMT_PAYMENT to see if theyre last loan was not accepted or cancelled
- Convert the DAYS variables on indicators that a loan was not accepted and count how many
- Product Combination dropped because it is redundant and introduces a very high cardinality on the data
- Fill AMT CREDIT with the median when the partitions are made

# bureau.csv
- Maybe drop the features that imply feature information (relative date > 0), since on the moment deciding we would not have that information
- Using the days credit enddate and active determine if the loan is overdue