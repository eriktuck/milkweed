# TODO

REMOVE DEVICE UUID FROM CONFIG AND EXTRACT-MM.ipynb

-   [x] Must allow for editing of previous transactions in Monarch and quick transfer to app.
-   [ ] Flip color of bar for savings, change color of annotation text
-   [ ] Title Case custom categories for easier plotting
-   [ ] Update color scheme for trends chart (use info, primary, purple, ...)
-   [ ] Auto-order CSP subcategories alphabetically (avoid need for CSP order in config)
-   [ ] See utilization-report-2 components.utils.no_matching_data() for how to replace the figure with "No Matching Data" message when empty.
-   [ ] See utilization-report-2 components.table_highlights for how to possibly improve transactions table formating
-   [ ] Add 'Household' as use case for trends
-   [ ] Add 'Household' as option for Actuals (skip the account filter and sum budget for all users) and Budget (non-editable summary)
-   [ ] Add callback for editing CSP table, refactor code
-   [ ] Update budget so that you can apply last year's spending month-by-month or input a value to broadcast across the entire year
    -   On click of category: Show small popup with line chart with last year, this year. Button on bottom that says apply last year's budget or numeric input with button that says broadcast to year.
-   [ ] Change CSP so that it only shows percents in the top table, and by clicking on a category, a second table will show the budgets or actuals in that category. Use hidden rows. 
-   [ ] Add joint_contribution to CSP in Income for Joint and in Fixed for both personal. 
-   [ ] Add reset switch for projections to switch to budget mid-scenario



Design Inspo

[HomeBot](https://buyers.homebotapp.com/reports/c0a0ff35-8753-4564-b1eb-618be27169db/home/599f4b0d0be89a00038d3c05#principal_interest)

### Next Up

-   [ ] Work with GPT to deploy to Cloud Run



-   Add in tax burden to withdrawals so we can cover expenses plus taxes
-   Add in RMD checks (no switching logic yet, just flag when RMDs are not met)
-   Add a portfolio for the house and allocate business net revenue as income?



https://huggingface.co/kuro-08/bert-transaction-categorization



Goals

-   Use Cursor and/or Windsurf and/or Claude Code to support development and learn these tools
-   Consider splitting out the models to a separate project (maybe coast-fire) than can be imported into milkweed and a separate project that is public facing. GPT how to accomplish something like this. 
-   Re-take Git course to refresh on Git especially advanced topics like branching to limit damage of agentic coders. 
-   Play Git game to further improve Git skills.



-   Add method to household to get contributions. Option to use net cash flow or proportion of total income (15% is what Ramit suggests). The portfolio should include both members' accounts; contributions will be made proportionally. 
-   Add IRA: split post-tax contributions to IRA if gross household income is under threshold
-   Asp: parse contributions by entity. Distribute business income based on ownership each year and/or create portfolio for joint entity (like our joint brokerage in Schwab).
-   Withdrawal: build out withdrawal plan that includes RMD and social securit
-   

FOR TXN CLASSIFIER: USE LORA TO TRAIN THE OUTPUT LAYER OF BERT (see LLM Engineering week 7 day 1).

## Debugging joint contribution

-   Something is wonky with the airbnb business. If I set the end date before coast years (e.g, 2034, 2033), it creates a huge spike in net cashflow, actually greater than it's revenue. 
    -   The tax burden decreases disproportionately with the drop in income when the business closes (using 2033). This causes a spike in net cash flow between close of business (2033) and coast (2036). Tax burden stays low throughout coast for some reason. Adjusted gross income looks accurate.  Combined expenses also looks accurate. So it's compute_taxes.
    -   The issue was how I added business taxes, it truncated at the intersection, meaning I had 0 federal wages after the business closed.
    -   MORAL OF THE STORY: YOU COPY PASTED CODE FROM CHATGPT WITHOUT PROPER TESTS AND EVERYTHING BLEW UP BUT YOU DIDN'T KNOW WHY!!!!

The difference is in how the more robust household.compute_taxes function works compared to the basic calc_total_taxes helper function in the household.get_joint_contribution function. The difference is approximately 14,073. Below, income_required is calculated from the latter function.

```python
income_required.loc[erik.get_coast_years()].apply(calculate_total_taxes) - household.compute_taxes().loc[erik.get_coast_years()]
>>> 14073.55

household.compute_net_cashflow().loc[erik.get_coast_years()]
>>> 14073.26
```



Income required is also almost exactly equal to the sum of erik.get_gross_income and rachel.get_gross_income

```python
income_required.loc[erik.get_coast_years()]
>>> 241996.53...

rachel.personal_income.get_federal_wages().loc[erik.get_coast_years()] + erik.personal_income.get_fica_wages().loc[erik.get_coast_years()]
>>> 241996.53
```



As a final check, expenses also add up

```python
individual_expenses = [person.personal_income.get_personal_expenses() for person in household.members]
joint_expenses = [expense.get_stream_series() for expense in household.expenses.values()]
business_expenses = [business.get_total_expenses() for business in household.businesses.values()]
health_care_expenses = [member.health_care.get_health_costs() for member in household.members]
expense_series = individual_expenses + joint_expenses + business_expenses + health_care_expenses

household.get_combined_expenses().loc[erik.get_coast_years()] == pd.concat(expense_series, axis=1).loc[erik.get_coast_years()].sum(axis=1)
```





-   Testing: Create a suite of performance tests to ensure that the full model returns the expected results for incomes and expenses, taxes, net cashflow, etc. using dummy data

-   Double check: Add a method for checking whether all expenses and incomes in a Transactions object are accounted for in the entity. Right now, we've done this explicitly by using csp_labels as primary filters. However if we don't add something to an entity explcitly it could be an issue. Instead sum all incomes and sum all expenses and then compare to get_total_expenses and get_total_incomes for the object. OR invert every filter function (maybe create a copy and delete transactions, then return the remaining transactions so you can see exactly what's missing). Do lazily to avoid running it everytime an expense is added. We only need to ensure we initialize the entity properly the first time.







Asp: create Home and Healthcare classes







When paychecks are greater than 0, contribute to 401(k) and HSA

When post-tax contributions are greater than 0, contribute to Roth IRA, 401(k), then taxable accounts



Contribution strategy

-   401(k) up to match
-   HSA up to max
-   Roth IRA (if under income limit)
-   401(k) up to max OR taxable **(empirical, allow user to test scenarios)**
    -   401(k) is better if delaying retirement AND/OR roth conversion is possible during coast
    -   taxable is better if roth conversion not possible
-   taxable
    -   distribute proportionally (simple) or according to glide path targets using greedy approach



Working years

-   Income
    -   Paychecks
    -   Business Income
-   Contributions
    -   401k (Income)
    -   HSA
    -   Adjusted Gross Income (Income, 401k, HSA, Deductions)
    -   Modified AGI (AGI, Deductions)
    -   Roth IRA (MAGI)
    -   Taxable
-   Expenses
-   



Coast years

-   Expenses
-   Taxes
-   Income (Expenses, Taxes): Guess-and-check problem
-   Strategies
    -   Roth Conversion



Retirement Years

-   Expenses
-   Withdrawals
-   RMD
-   Taxes (Expenses, Withdrawals)
-   Strategies





Retirement planning

-   expand contribution plan to include 401(k) by gross income for all years working (default to 3% plus match but allow for extra up to max); 
-   expand contribution plan to take as input a single value for post-tax and to redirect from Roth to taxable accounts if income exceeds legal limit for contribution
-   update projections to take as input a series of contributions
-   update projections to work for entire portfolio



Design trends page:

-   Study Monarch reports to understand what works and what doesn't
-   Draft questions to answer
-   Design charts
-   Design layout and UI

Add click through drill down for category_name in trends table



Build Airbnb page:

-   Copy uploader widget from Actuals 
-   Parse airbnb data, optionally save to disk
-   Create layout with uploader as UI, include instructions from Notion on how to get reports
-   Convert seaborn plots to plotly plots
-   Write callbacks to update all charts on load (from disk) or upload



Build investment page: 

table for inputting investment accounts, assigning category (retirement/investment), assign user (erik, joint, rachel)



Investments should show total net worth, include checking and savings (cash), non-retirement accounts, retirement accounts (401k, Roth IRA), crypto, assets (house, car), and debt (mortgage). 

-   What is the projected total net worth at retirement? Are we saving enough, too much? (this bleeds into the Retirement page but I want the Retirement page to focus on retirement income)
-   How large will each account type be at key milestones (e.g., at 55, how much will we have in liquid assets)
-   Asset allocations by risk (see Personal Capital) to support rebalancing.
-   Asset allocations by sector to support rebalancing. Include benchmarks. 
-   Rebalancing module. Create module to automatically suggest changes in allocations (with buy orders, not sell orders) over the year to rebalance.
-   Table of account values by account type, account, and holder
-   Table of transactions (actual)
-   Simulation of 







Build retirement page:

-   Design a visual that reduces the complexity of retirement decisions, good l



-   what is the tax bleed of each scenario? (Required Minimum Distributions, Social Security Taxes)
-   When to buy annuity to maintain standard of living through late retirement?
-   How does housing factor in?
-   Social security will vary between \$1965 (if no future income earned) to \$4682 if earning an average of 
    -   Average indexed monthly earnings during the 35 years in which you earned the most



Retirement variables

-   Current assets (calculate from provided)
-   Ages (slider with non-overlapping ranges)
    -   Coast
    -   Retirement
    -   Social Security
    -   Slow-go years
    -   No-go years
-   Death age (use 90, but doesn't matter with annuity assumption)
-   Inflations
    -   Income inflation factor (social security benefits, contribution inflation)
    -   Fixed spending inflation factor during working years
    -   Guilt free spending inflation factor during working years
    -   Contribution inflation factor (determined by income inflation less spending inflation)
    -   Consumer Price Inflation
-   Returns (allow user to increase the conservativism of the estimate).
    -   Returns on equity
    -   Returns on fixed income
-   Portfolio allocation (cash, bonds and other fixed income, equities, crypto; prop in retirement and non-retirement accounts will affect max and min distributions at each age)
-   Tax rates (social security tax rate, income tax rate, capital gains tax rate)
-   Home value, appreciation and pay off date. 
    -   Decision to sell (invest profits) and rent OR hold and essentially reduce housing cost to 0
-   Housing costs (before mortgage paid, after mortgage paid)
-   Health Insurance costs (earnings years, coast years, medicare at 65, long-term care)
-   Retirement spending 
    -   (housing, insurance, fixed costs, guilt-free spending; fluctuates in go-go, slow-go, and no-go years)
-   Age to buy annuity
-   



### Total Net Worth is a Linear combination of arrays with nested dependencies

Earned income to coast (earned income after coast is equal to expenses based on CoastFIRE assumptions)

Health expenses

Mortgage/Rent

Other fixed expenses

Guilt-free spending



Expenses (health expenses, housing expenses, other fixed expenses, guilt-free spending)

Earned income after coast (expenses)

Earned income (earned income to coast, earned income after coast)

Income taxes (Earned income)

Social security benefit (Earned income)

Earned income and ss benefit (earned income, ss benefit)

**Income less expenses** (Earned income and ss benefit, expenses) <- to say $0 contribution, move coast up to current year



Withdrawals (income less expenses) <- not to reach 0, just to cover expenses, may result in leftover money after death. you optimize with sliders and numeric inputs.

Withdrawals by account type (withdrawls, allocations) <- comes after knowing withdrawals based on most tax-efficient strategy

Taxable withdrawals (Withdrawals by account type)

Non-taxed withdrawals (Withdrawls by account type)







Total income (earned income, social security income, withdrawals)

Social security taxes (earned income, social security income, taxable withdrawals)

Capital Gains taxes (earned income, withdrawals by account type)

Total taxes (income taxes, social security taxes, capital gains taxes)







## CSP Philosophy

The power of the CSP is quickly evaluating simple scenarios: How much does Erik need to make to hit savings targets? Can we afford a maid service? How much should we spend on vacation this year? Can we afford that home project? These questions are near-term (one year or less) and involve tradeoff decisions on spending.

The CSP can't answer questions that involve long-term scenario planning like: How long will our money last if I don't get a job as soon as planned? It's also too rough to answer questions about how much we should invest to retire early (it only offers rough percent ranges suggestions).

To replace the IWT Spreadsheet, we'll need an editable table that can allow the same simple scenario testing. The improvement from a simple spreadsheet will be (1) auto-loading from sources such as actuals or budgets and (2) reporting back the difference between the scenario and the current plan (or auto-generating a budget from the plan).

-   Load values from any previous or current year's actual spending (except for Rachel's, leave blank)
    -   Average from 1 to 12 months
-   Load values from any previous or current year's budget (get Rachel from previous CSP Sheets)
    -   Averaged from 1 to 12 months
-   User edits table values
-   Dash sums values in total column (can do pinned column? or need to blow away grid on each edit? or use save button to allow undoing? or a separate table with rows for each user like a pivot table?)
-   Save out to config? Or report differences from source? Or save out as new budget?

The CSP will still need to be augmented with 

-   Investment amount required to hit retirement goal (Retirement -> Budget)

A totally different module will be needed for retirement scenario planning, but the CSP could provide a useful view on

-   Retirement scenarios with no mortgage, or lower cost of living, etc. (given a retirement income, what can we spend on housing, travel, etc?)

## Decisions informed

-   Retirement planning
    -   Investing target (per person)
-   Annual planning
    -   Joint contribution amount
    -   Investing target (per person)
    -   Vacation budget
    -   Home budget
-   Monthly
    -   Confirm joint spending is on track
    -   Confirm personal spending is on track
-   Daily
    -   Afford purchases

### 2024-01-10

Designed a more thorough workflow for both local and web deployment. Moved the use case selector to a "configuration bar" under the nav bar and migrated code to config.py. Moved navbar to app.py. Built functionality to calculate difference between total income and total expenses

### 2024-01-09

Built the budget page with dash AG data table. Added a sum row on edit of cell value.

### 2024-01-08

Frustrated by a lack of clarity in design. Went in circles on the chart, no real progress.

### 2024-01-07

The data flow is slightly different with this version as the transactions need to be processed uniquely for each user. Storing the transactions in a session exceeded the storage limits, but I can still use [dcc.Store](https://dash.plotly.com/dash-core-components/store) with a memory storage but we'll see if that is doable with multi-page since it is lost on page reload. I may need to fall back on the file system and figure something else out for deployment.

### Conditions of Satisfaction v1.0

-   Budget chart for both 
-   CSP chart for both Erik and Joint



Data flow

Transactions should be raw from Monarch, they will require user-specific transformation. This is different than the original finance app where the transactions could be stored as is and simply filtered. However, we don't want to re-process with each date filter, so we will stage all data in transactions-data-store and then date-filtered in subsetted-transactions-store.

1.   User selects use case ('erik', 'joint')
2.   run callback store_config (triggered by use-case): Dash reads config file from disk and stores with dcc.Store (id=config-store)
3.   run callback read_transactions (triggered by config-store): Dash reads transactions data from disk and processes based on config file and stores with dcc.Store (id=transaction-data-store)
4.   run callback store_subsetted_transactions (triggered by transasction-data-store or datapickers): Dash subsets transactions by date and stores with dcc.Store (id=transaction_subset_store) TODO: this could be eliminated if updating the table is done with full transactions dataset
5.   run callback update_plot (triggered by transaction-subset-store): Dash displays the budget chart

User updates use case:

1.   Re-run all steps above

User updates date:

1.   run callback store_subsetted_transactions

User runs fetch

1.   Login (if required; save session to disk), select dates, and fetch transactions
2.   Update existing transactions with new transactions and store in dcc.Store (id=transaction-data-store). Optionally save to disk.
3.   Re-run callback store_subsetted_transactions to process updated transactions based on config file. Re-run update plot.

## Purpose

- Decide whether I can afford (relative to my budget) a $300+ purchase
- Confirm my annual spending to date is within the range of my estimated budget
- Confirm joint annual spending to date is within the range of our estimated budget
- Review CSP category spending over time (at top level: fixed, savings, guilt-free; as both percent of income and total amount)
- Confirm spending to date (especially guilt-free spending and savings goals: vacation, home improvements) are within parameters

## Design

A multi-page Dash app, ideally supporting multiple users.

-   Planned versus Actual: shows budgeted spending, for any time period, based on use case
-   Budget: table to create and edit budgets for any time period
-   CSP: a page for reviewing spending by CSP category and trying new values (could this interact with budget?)
-   Trends: a multi-visual dashboard for interrogating spending patterns
-   Investments: ensure investments are balanced, test new allocations, input transactions
-   Crypto: review crypto performance over time
-   Retirement: a retirement calculator using data from budget
-   Airbnb: airbnb analysis (from file, no API)

### Users

For now, Erik is the only anticipated user. The raw-transactions file and monarch money session are read from (and saved to) disk.

The config file has global configuration settings which include a list of groups and categories from MonarchMoney and, under "users", a dictionary of use cases (see below).

### Use cases

A single user may have multiple use cases. For example, I have both a personal view and a joint view. To see the total for the household I also need to track basics about Rachel, including her annual budget.

Users can customize use case configuration settings including custom mapping of the transactions categories to CSP categories, and assignment of account owners (TODO). 

If the app is to be deployed for other's use, the handling of transactions and config will need to be updated so that they are stored for specific users in a database or uploaded when loading the app for the first time. This latter option could be accomplished by launching a modal on load to request the raw-transactions and config file.



## Workflow





### On Launch (for web app)

App checks for mm object, config file and transactions_raw. If not present (e.g., app is launched from web app not locally), launch login dialog box:

### Login dialog box

-   If user has used app before, request zipped config file and offer option to close or launch transactions dialog box, else:
-   Button to connect to Monarch
-   Get categories and groups
-   -> Save to config in dcc.Store
-   -> Launch category management widget

### Category management widget (per use case)

-   Read categories from config
-   User inputs use case name (for editing, read from config file or allow new)
-   User assigns CSP category to each Monarch category
-   User assigns CSP label to each CSP category (must use all of Income, Fixed Costs, Investments, Savings; guilt-free is everything else)
-   User selects Monarch categories to drop
-   User assigns accounts to owners (i.e., use cases)
-   -> Update config
-   -> Launch update transactions dialog box

### Update transactions dialog box

-   Read categories from config
-   Date picker range for date range of transactions to pull
-   Button to download (should launch some username, password in browser if not already stored)
-   Delete all transactions within date range from transactions_raw and concat replacements
-   





User will download their data for the next use (mm object, config, transactions) to avoid re-doing the login process





-   Use Material launcher to launch dialog box for user, user selects start/end date, presses button to request data by date (no other options at this time) 
-   Data are transformed and saved to Config file (JSON format)
-   User assigns categories to CSP creates budget based on transaction categories



Use API to get stock data
- https://yahooquery.dpguthrie.com/
- mstarpy
- investpy
- https://github.com/ranaroussi/yfinance
- https://pythoninvest.com/long-read/exploring-finance-apis
- https://github.com/Nasdaq/data-link-python?tab=readme-ov-file

See this article for how to get sector info from yfinance: https://wire.insiderfinance.io/s-p-500-python-analysis-with-yfinance-plotly-and-pandas-d8fb3ca5830e



Longevity calculators recommended by Bill Perkins

https://www.livingto100.com/?mobile=0

https://www.longevityillustrator.org/



Consumer Price Inflation

Use `cpi` for consumer price inflation historicals

```python
!pip install cpi
import cpi
cpi.update() # Updates the CPI data to the latest available
cpi.get(year) # Retrieves the CPI for the specified year
```



In models.py, use cache rather than explicitly defined private variables to more easily clear cache





```python
class Household(FinancialEntity):
    def __init__(self, name, members, transactions=None, businesses=[], assets=[]):
        super().__init__(name, incomes=[], expenses=[])
        self.name = name
        self.members = members
        self.businesses = {business.name: business for business in businesses}
        self._cache = {}  # Add caching mechanism
        
    def _cache_key(self, method_name, **kwargs):
        """Generate a cache key based on method name and parameters"""
        return f"{method_name}_{hash(frozenset(kwargs.items()))}"
    
    def foobar(self, force_recalculate=False):
        """Cached version of combined income calculation"""
        cache_key = self._cache_key('combined_agi')
        
        if not force_recalculate and cache_key in self._cache:
            return self._cache[cache_key]
            
       # Otherwise calculate combined agi
            
        self._cache[cache_key] = result
        return result

    def invalidate_cache(self):
        """Clear cached calculations when data changes"""
        self._cache.clear()
```

