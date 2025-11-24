# from .business import Business
# from .household import Household
# from .individual import Individual, PreTaxContribution
# from .portfolio import Portfolio, Account, Holding
# from .core import Stream
# from .transactions import Transactions
# from .healthcare import HealthCare

# # if __name__ == 'main':
# print('v0.0.4')
# import pandas as pd
# import yfinance as yf
# import importlib
# import core.utils.functions
# importlib.reload(core.utils.functions)
# from core.utils import functions 

# def fixed_expense_filter(df):
#     """Filter for fixed expenses excluding_housing and health insurance."""
#     filt = (
#         (df['csp_label'] == 'fixed') & 
#         (df['csp'] != 'mortgage') &
#         (df['csp'] != 'airbnb') &
#         (df['csp'] != 'joint_contribution')
#     )
#     return df.loc[filt]

# def joint_contribution_filter(df):
#     filt = (
#         (df['csp'] == 'joint_contribution')
#     )
#     return df.loc[filt]

# def healthcare_expense_filter(df):
#     """Filter for healthcare (insurance)."""
#     filt = (
#         (df['csp'] == 'health_insurance')
#     )
#     return df.loc[filt]

# def mortgage_expense_filter(df):
#     filt = (
#         (df['csp'] == 'mortgage')
#     )
#     return df.loc[filt]

# def goals_expense_filter(df):
#     filt = (
#         (df['csp_label'] == 'savings')
#     )
#     return df.loc[filt]

# def discretionary_expense_filter(df):
#     """Filter for discretionary (guilt-free) expenses."""
#     filt = (
#         (df['csp_label'] == 'guilt-free')
#     )
#     return df.loc[filt]

# def airbnb_income_filter(df):
#     filt = (
#         (df['csp'] == 'income')
#     )
#     return df.loc[filt]

# def airbnb_expense_filter(df):
#     filt = (
#         (df['csp'] == 'airbnb')
#     )
#     return df.loc[filt]

# def airbnb_write_off_filter(df):
#     # Get prorated writeoffs and prorate
#     filt = (
#         ((df['csp'] == 'bills_utilities') |
#         (df['csp'] == 'television'))
#     )
#     df = df.loc[filt]
#     df.loc[:, 'amount'] = df.loc[:, 'amount'] * 0.25

#     return df

# # contribution limits 
# current_contribution_limits = pd.Series(
#     [
#         14000, 15000, 15500, 15500, 16500, 16500, 16500, 17000, 17500, 17500,
#         18000, 18000, 18000, 18500, 19000, 19500, 19500, 20500, 22500, 23000, 23500
#     ], 
#     index=range(2005, 2026)  # Index from 2005 to 2025
# )
# future_contribution_limits = pd.Series(current_contribution_limits.iloc[-1],
#                                     index=range(2026, 2100))
# contribution_limits = pd.concat([current_contribution_limits, future_contribution_limits])

# # Initialize Erik's Transactions
# erik_transactions = Transactions(name="erik")

# # Create Erik as an Individual with personal transactions
# erik = Individual(name="erik", birth_year=1986, 
#                   transactions=erik_transactions)

# # Assign past gross income (user-supplied)
# payroll = pd.read_csv('../payroll_data.csv', parse_dates=["Date"])
# gross_income = payroll.groupby(payroll['Date'].dt.year)['Gross Income'].sum()

# erik.personal_income.past_gross_income = gross_income

# # Add a manual raise in 2025
# erik.personal_income.add_manual_income_entry(2026, 175000)

# # Create Expense instances for Erik
# fixed_expense = Stream(
#     transactions=erik_transactions, name="Fixed", end_year=erik.death_year, 
#     filter_func=fixed_expense_filter, use_budget=True)
# joint_contribution = Stream(
#     transactions=erik_transactions, name="Joint Contribution", 
#     end_year=erik.death_year, 
#     filter_func=joint_contribution_filter, use_budget=True)
# discretionary_expense = Stream(
#     transactions=erik_transactions, name="Discretionary", end_year=erik.death_year, 
#     filter_func=discretionary_expense_filter, use_budget=True)
# goals_expense = Stream(
#     transactions=erik_transactions, name="Goals", end_year=erik.death_year, 
#     filter_func=goals_expense_filter, use_budget=True)

# # Add expenses to Erik
# erik.add_expense(fixed_expense)
# erik.add_expense(joint_contribution)
# erik.add_expense(discretionary_expense)
# erik.add_expense(goals_expense)

# # Add pre-tax deductions
# fzok = PreTaxContribution(name="401k", rate=0.05,
#                         max_contribution=contribution_limits,
#                         start_year=2004, end_year=2024)
# fzok2 = PreTaxContribution(name="401k", rate=0.05,
#                         max_contribution=contribution_limits,
#                         start_year=2026, end_year=erik.coast_year - 1)
# hsa = PreTaxContribution(name="HSA", rate=0.03,
#                         max_contribution=contribution_limits,
#                         start_year=2022, end_year=2024,
#                         matched=False)
# hsa2 = PreTaxContribution(name="HSA", rate=0.03,
#                         max_contribution=contribution_limits,
#                         start_year=2026, end_year=erik.coast_year - 1,
#                         matched=False)

# erik.personal_income.add_pre_tax_contribution(fzok)
# erik.personal_income.add_pre_tax_contribution(fzok2)
# erik.personal_income.add_pre_tax_contribution(hsa)
# erik.personal_income.add_pre_tax_contribution(hsa2)

# erik_healthcare = HealthCare(individual=erik, 
#                              employer_premium=0, 
#                              out_of_pocket=0, 
#                              aca_premium=550*12, 
#                              medicare_premium=500*12, 
#                              end_of_life_cost=50000)
# erik.assign_healthcare(erik_healthcare)

# # Initialize Rachel's Transactions
# rachel_transactions = Transactions(name="rachel")

# # Create Rachel as an Individual with personal transactions
# rachel = Individual(name="rachel", birth_year=1988, 
#                     transactions=rachel_transactions,
#                     coast_age=48)

# # Assign past gross income (user-supplied)
# earnings_years = [2024]
# earnings = [175000]
# previous_income = pd.Series(earnings, index=earnings_years)

# rachel.personal_income.past_gross_income = previous_income

# # Create Expense instances for rachel
# fixed_expense = Stream(
#     transactions=rachel_transactions, name="Fixed", end_year=rachel.death_year, 
#     filter_func=fixed_expense_filter, use_budget=True)
# joint_contribution = Stream(
#     transactions=rachel_transactions, name="Joint Contribution", 
#     end_year=rachel.death_year, 
#     filter_func=joint_contribution_filter, use_budget=True)
# discretionary_expense = Stream(
#     transactions=rachel_transactions, name="Discretionary", end_year=rachel.death_year, 
#     filter_func=discretionary_expense_filter, use_budget=True)
# goals_expense = Stream(
#     transactions=rachel_transactions, name="Goals", end_year=rachel.death_year, 
#     filter_func=goals_expense_filter, use_budget=True)

# # Add expenses to rachel
# rachel.add_expense(fixed_expense)
# rachel.add_expense(joint_contribution)
# rachel.add_expense(discretionary_expense)
# rachel.add_expense(goals_expense)

# # Add pre-tax deductions
# fzok = PreTaxContribution(name="401k", rate=0.03,
#                         max_contribution=contribution_limits,
#                         start_year=2024, end_year=rachel.coast_year - 1)
# hsa = PreTaxContribution(name="HSA", rate=0.03,
#                         max_contribution=contribution_limits,
#                         start_year=2024, end_year=rachel.coast_year - 1,
#                         matched=False)

# rachel.personal_income.add_pre_tax_contribution(fzok)
# rachel.personal_income.add_pre_tax_contribution(hsa)

# rachel_healthcare = HealthCare(individual=rachel, 
#                              employer_premium=0, 
#                              out_of_pocket=0, 
#                              aca_premium=550*12, 
#                              medicare_premium=500*12, 
#                              end_of_life_cost=50000)
# rachel.assign_healthcare(rachel_healthcare)

# # Initialize Household Transactions
# joint_transactions = Transactions("joint")

# # Create Household
# household = Household(name="joint", members=[erik, rachel], 
#                     transactions=joint_transactions)

# # Create Expense instances for Household
# fixed_expense = Stream(
#     transactions=joint_transactions, name="Fixed", 
#     end_year=household.end_year, 
#     filter_func=fixed_expense_filter, use_budget=True
# )
# discretionary_expense = Stream(
#     transactions=joint_transactions, name="Discretionary", 
#     end_year=household.end_year, 
#     filter_func=discretionary_expense_filter, use_budget=True
# )
# mortgage_expenses = Stream(
#     transactions=joint_transactions, name="Mortgage",
#     end_year=2052,
#     filter_func=mortgage_expense_filter, use_budget=True
# )
# healthcare_expenses = Stream(
#     transactions=joint_transactions, name="Healthcare",
#     end_year=household.end_year,
#     filter_func=healthcare_expense_filter, use_budget=True
# )
# goals_expenses = Stream(
#     transactions=joint_transactions, name="Goals",
#     end_year=household.end_year,
#     filter_func=goals_expense_filter, use_budget=True
# )

# household.add_expense(fixed_expense)
# household.add_expense(discretionary_expense)
# household.add_expense(mortgage_expenses)
# # household.add_expense(healthcare_expenses)
# household.add_expense(goals_expenses)

# # Add income to household
# joint_contribution = Stream(
#     transactions=joint_transactions, name="Joint Contribution",
#     end_year=household.end_year, 
#     filter_func=joint_contribution_filter,
#     use_budget=True
# )
# household.add_income(joint_contribution)


# # Initialize Airbnb Business
# ownership = {household: 1.0}
# airbnb = Business(
#     name="Airbnb",
#     exit_year=2033,
#     transactions=joint_transactions,
#     ownership=ownership
# )

# airbnb_income = Stream(transactions=joint_transactions, 
#                     name="Business Income", 
#                     end_year = airbnb.exit_year,
#                     filter_func=airbnb_income_filter)

# airbnb_expenses = Stream(transactions=joint_transactions, 
#                         name="Airbnb", 
#                         end_year=airbnb.exit_year, 
#                         filter_func=airbnb_expense_filter)

# airbnb_write_offs = Stream(transactions=joint_transactions, 
#                         name="Airbnb Writeoff", 
#                         end_year=airbnb.exit_year, 
#                         filter_func=airbnb_write_off_filter)

# airbnb.add_income(airbnb_income)
# airbnb.add_expense(airbnb_expenses)
# airbnb.add_write_off(airbnb_write_offs)

# # Add business to household
# # household.add_business(airbnb)

# # Assign joint contribution amounts
# household.assign_joint_contributions()

# # Compute taxes
# household.compute_taxes()

# # Assign taxes
# household.assign_allocated_taxes()

# # Create portfolio
# erik_portfolio = Portfolio()

# # Read holdings from csv
# erik_holdings = pd.read_csv('../data/mw_holdings.csv')
# tickers = list(erik_holdings['symbol'].unique())
# multi_data = yf.Tickers(tickers)

# # Get cost basis from Vanguard report
# brokerage_370_basis = functions.load_vanguard_cost_basis('../notebooks/data/costbasisdownload_1370.csv')
# brokerage_370_basis['account'] = 'Vanguard Brokerage ...370'

# brokerage_191_basis = functions.load_vanguard_cost_basis('../notebooks/data/costbasisdownload_8191.csv')
# brokerage_191_basis['account'] = 'Vanguard Brokerage ...191'

# cost_basis = pd.concat([brokerage_370_basis, brokerage_191_basis])
# erik_holdings = erik_holdings.merge(cost_basis, on=['account', 'symbol'], how='left')

# # Add accounts and holdings to portfolio
# for account in erik_holdings['account'].unique():
#     filt = erik_holdings['account'] == account
#     holdings = erik_holdings.loc[filt]
    
#     account_type = holdings['account_type'].iloc[0]
#     account = Account(account, account_type, holdings=None)

#     for idx, holding in holdings.iterrows():
#         symbol = holding['symbol']
#         shares = holding['shares']
#         ticker_obj = multi_data.tickers[symbol]
#         cost_basis = holding['cost_basis']

#         holding = Holding(symbol, shares, ticker_obj, cost_basis)
#         account.add_holding(holding)

#     erik_portfolio.add_account(account)

# erik.personal_income.portfolio = erik_portfolio
