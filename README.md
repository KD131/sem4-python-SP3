# SP 3 - Diabetes
*by gruppEN (Frederik, Robert, Rasmus, Janus)*

Datasæt: sklearn.datasets.load_diabetes()

Only use 2 features: bmi & bp
* bmi body mass index
* bp average blood pressure

1. Perform linear regression where x = bmi and y = bp
2. Show the regression equation (y=ax+b)
3. Show the regression line on a scatterplot with the other datapoints.
4. According to your model, what is y when x=2.

# Status

**TBD**

To run it, just execute the python file like so: `python rotten_tomatoes.py`

If the CSV files `movies_year.csv` are not present in the folder, it will attempt to scrape the data and save it to CSV. You can also force this behaviour by passing `refresh=True` to the `get_movies()` function.

The script will show a plot of the average audience rating of each genre for 2010 and 2020. After closing the plot window, it will then print the average runtime of dramas for 2010 and 2020.
