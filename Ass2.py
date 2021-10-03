## Assignment 2 code submission

import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import pearsonr
from sklearn import linear_model
import numpy as np

##########################################################################################
def get_data_income():
    """ 
    Pull data from 'Total_Income_by_GCCSA.csv' file
    (As we are getting data for Victoria, since all the values for Victoria is the same 
    throughout the Total_Income csv files, we just picked one file)
    
    Collect the data for just Victoria on the Median and Mean values from 2011 to 2018
    
    return dataframe of the result
        - rows = 1 = All data regarding Victoria
        - columns = 14 = [(Median ($), 2011/12 - 2017/18), (Mean ($), 2011/12 - 2017/18)]
    """
    
    # Get data with the first two rows as headers
    # Does not matter which data we use as we will only look at Victoria and it is the 
    #  same values for Victoria in every dataset
    income_df = pd.read_csv('Total_Income_by_GCCSA.csv', header = [0,1])
    
    # We only will be working with mean and median of total income
    income_VIC = income_df.loc[income_df[("GCCSA", "GCCSA")] == 
                               "Victoria",["Median ($)", "Mean ($)"]
                              ].reset_index(drop = True) 

    return income_VIC


###########################################################################################
def get_data_emission():
    """ 
    Pull data from 'Emissions(Air).csv' file
    
    Collect data for just Victoria on air emissions (Sulfur Dioxide only) from 2011 to 2018
    
    return dataframe of result
        - rows = 7 = According to each year
        - columns = 2 = [report year, Sulfur Dioxide total emission (kg)]
    """
    
    years = []
    for i in range(2011, 2018):
        years.append(str(i) + '/' + str(i+1))
    
    # Get data with the first row as header
    emission_df = pd.read_csv('Emissions(Air).csv', header = 0)
    
    # We only will be working with Sulfur Dioxide emissions in Victoria from 2011-2018
    so2_VIC = emission_df.loc[(emission_df["state"] == "VIC") 
                              & (emission_df["substance_name"] == "Sulfur dioxide") 
                              & (emission_df["report_year"].isin(years))
                              , ["report_year", "air_total_emission_kg"]]

    # Now we have to group all the sulfur dioxide per report year together
    so2_VIC_perYear = so2_VIC.groupby(["report_year"]).agg(so2_total_emission_kg = 
                                                           pd.NamedAgg(column = "air_total_emission_kg", 
                                                                       aggfunc = "sum")
                                                          ).reset_index() # Get all the column headings in place
    
    return so2_VIC_perYear


###########################################################################################
def scatterplot(income_VIC, so2_VIC, measure):
    """ 
    Create scatterplot of mean/median of income and sulfur dioxide emissions in Victoria
    from 2011-2018
    
    Get the pearson correlation of income and sulfur dioxide emissions
    
        The pearson correlation(r) lies within [-1,1]:
         - +1 --> perfect positive linear correlation
         - -1 --> perfect negative linear correlation
         - 0 --> no correlation
         - |r| --> strength of linear correlation
    
    Outputs a png file with the scatterplot of desired mean or median measure
        - together with pearson coefficient
    
    Returns values for emission and income in each separate lists for 
     each year(2011-2018)
    """
    
    incomes = []
    emissions = []
    
    # Plot the point in scatterplot for each location
    for index, row in so2_VIC.iterrows():
        
        # Change year format from so2_VIC df to income_VIC format
        year = row["report_year"][:4] + "-" + str(int(row["report_year"][2:4])+1)
        
        # Get the income as integers
        income = int(re.sub(",", "", income_VIC[(measure + " ($)", year)][0]))
        
        plt.scatter( income , row["so2_total_emission_kg"]/1000000)
        
        # Add texts to each data points to track which is which
        plt.annotate(year, (income, row["so2_total_emission_kg"]/1000000), 
                     textcoords="offset points", xytext=(5,3), ha='center')
        
        # Add the point values to respective lists
        incomes.append(income)
        emissions.append(row["so2_total_emission_kg"])
    
    # Get the pearson correlation
    r = pearsonr(incomes, emissions)[0]
    
    #Plot scatterplot for income vs emissions
    plt.grid(True)
    plt.xlabel(measure + " Income Per Year ($)")
    plt.ylabel("Sulfer Dioxide Emissions Per Year (kg) in millions")
    plt.title(measure + 
              " Income and Sulfer Dioxide Emissions Per Year in Victoria\nPearson Correlation = {:.4f}".format(r), 
              pad = 15)
    
    plt.savefig(measure + '_scatter.png')
    
    # Clear plot to create the next one
    plt.clf()
    
    return incomes, emissions, r


###########################################################################################
def linear_reg(x, y, measure):
    """
    Do regression analysis given two sets of data
         - Linear Regression used
         - Find the equation (Use it to compare with scatterplots)
         - Find the coefficient of determination to see how much variation y is explained
             by variation in x
    
    Plot residuals and output to visualise to check assumptions of Linear Regression
        - Linearity
        - Independence of residuals
    
    Outputs a png file with the residual plot
    """
    
    # Find the regression line
    lm = linear_model.LinearRegression()
    lm.fit(np.asarray(x).reshape(-1,1), np.asarray(y))
    equation = "SO2_Emissions = {:,.2f} + ({:,.2f}){}_Income".format(lm.intercept_,
                                                                    lm.coef_[0],
                                                                    measure)
    
    # Find Coefficient of Dettermination(R^2)
    r2 = lm.score(np.asarray(x).reshape(-1,1), np.asarray(y))
    
    # Plot residual plot
    predictions = lm.predict(np.asarray(x).reshape(-1,1))
    residual = [(y - yh)/1000000 for y, yh in zip(y, predictions)]
    plt.ylim([-50,50])
    plt.scatter(predictions, residual, color = 'C0', label = "R^2: {0:.2f}".format(r2))
    plt.plot([min(predictions), max(predictions)], [0,0], color= 'C2')
    plt.legend()
    plt.ylabel("in millions")
    plt.title("Residual Plot\n{}".format(equation), pad = 15)
    
    plt.savefig(measure + ' Residual Plot.png')
    
    # Clear plot to create the next one
    plt.clf()

    
###########################################################################################
def line_chart(points, name):
    """
    Get the line charts of individual variables (variable (y-axis), years(x-axis))
    
    Outputs a png file with desired variable by years
    """
    
    years = ["2011/12", "2012/13", "2013/14", "2014/15", "2015/16", "2016/17", "2017/18"]
    
    plt.plot(years, points)
    plt.title(name + " in Victoria")
    plt.xlabel("Year")
    
    # Get the units inside respectively
    if (name == "SO2 emissions"):
        plt.ylabel(name + " (kg)")
    else:
        plt.ylabel(name + " ($)")
    
    plt.savefig(name + ".png")
    
    # Clear plot to create the next one
    plt.clf()


###########################################################################################
def line_chart_2(points, name, pearson):
    """
    Get line chart comparing two lines in the same chart
      - 2 y-variables(sulfur dioxide emissions vs Mean/Median Income)
      - 1 x-variable(Years)
      
    Outputs a png file with (desired mean or median measure + emissions) by years
      - together with pearson coefficient
    """
    
    years = ["2011/12", "2012/13", "2013/14", "2014/15", "2015/16", "2016/17", "2017/18"]
    
    plt.plot(years, points, "g", label = name)
    plt.xlabel("Year")
    plt.ylabel(name + " Income ($)", color = 'green')
    
    # creating the second line graph in the same plot
    p2 = plt.twinx()
    p2.plot(years, mean_SO2, label = "SO2 emissions")
    p2.set_ylabel("SO2 emissions (kg)", color = 'blue')
    
    plt.title(name + " Income and Sulfur Dioxide Emissions Per Year in Victoria\nPearson Correlation = {:.4f}".format(pearson))
    plt.grid()
    
    plt.savefig(name + ".png")
    
    # Clear plot to create the next one
    plt.clf()
    
###########################################################################################
if __name__ == '__main__':
    
    # Get the dataset for total income from 2011-18 (Mean and Median) for Victoria
    income_VIC = get_data_income()

    # Get the dataset for Emissions from air(Sulfar Dioxide) for Victoria
    so2_VIC = get_data_emission()
    
    # Get scatterplot for median and mean income vs Sulfur Dioxide emission
    # + pearson correlation
    med_income, med_SO2, med_r = scatterplot(income_VIC, so2_VIC, "Median")
    mean_income, mean_SO2, mean_r = scatterplot(income_VIC, so2_VIC, "Mean")
    
    # Do regression analysis
    linear_reg(med_income, med_SO2, "Median")
    linear_reg(mean_income, mean_SO2, "Mean")
    
    # Get the line chart of mean/median income and emissions by years individually
    line_chart(med_income, "Median Income")
    line_chart(mean_income, "Mean Income")
    line_chart(mean_SO2, "SO2 emissions")
    
    # Get the line chart of mean/median income and emissions by years together
    line_chart_2(med_income, "Median", med_r)
    line_chart_2(mean_income, "Mean", mean_r)