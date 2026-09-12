# DECISIONS TAKEN
Here are the Decisions taken on the project, with the objective to have it all documented and no decision is made out ouf nowhere or unexplained

# application_train.csv
- Treat ir first so it is the main table, do basic stuff to clean data, quit all the NA, basic feature encoding, basic stuff
- Leave Kaprimont to see if different categories do affect the target or they are all basically the same
# installments_payments.csv
- First explore it, because it is historical data about other loans and not the actual loan there is no leakage as we are not using information about the state of the actual loan, and all the information is not mixed on the temporal dimension with the actual loans, but first check it
- I could make variables with this about relative fully repay date, if someone payed on date, before or after, if they have a loan active right now, percentage of total amount repayed, average days took to repay it fully, maximum days it took to repay it across all the loans and a metric determining if on each loan they're behavior is getting better or worse, this is they pay sooner or later or they dont pay at all
- I could extract out of total loans how many have they fully repayed, maybe a proportion of loans_repayed/total_loans or money_repayed/total_money_topay