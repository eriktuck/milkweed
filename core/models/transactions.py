import json
import datetime
import pandas as pd
import core.utils.functions as functions


class Transactions:
    def __init__(self, name):
        self.name = name  # Used for transaction/budget filtering
        self._transactions = None
        self._aggregated_transactions = None
        self._budget_transactions = None

    def _get_transactions(self):
        if self._transactions is None:
            raw_transactions = pd.read_pickle('../data/raw-transactions.pkl')
            with open('../data/config.json', 'r') as f:
                config = json.load(f)
        
            # Read transactions
            self._transactions = functions.categorize_transactions(
                raw_transactions, config, self.name
                )

        return self._transactions
    
    def _get_budget(self):
        if self._budget_transactions is None:
            with open('../data/config.json', 'r') as f:
                config = json.load(f)
            user_config = config["users"][self.name]
            csp_labels = user_config['csp_labels']
            budget_json = user_config.get('budget', {})
        
            budget = []
            for year, months in budget_json.items():
                for month, categories in months.items():
                    for category, amount in categories.items():
                        budget.append(
                            {"year": int(year), 
                             "month": int(month), 
                             "csp": category, 
                             "amount": amount}
                        )
                        
            budget_df = pd.DataFrame(budget)
            budget_df['date'] = pd.to_datetime(
                budget_df['year'].astype(str) 
                + '-' + budget_df['month'].astype(str) 
                + '-01'
            )
            budget_df = budget_df.assign(
                csp_label=lambda x: x['csp'].map(csp_labels)
            )
            filt = budget_df['csp_label'] != 'income'
            budget_df.loc[filt, 'amount'] *= -1
            self._budget_transactions = budget_df
        return self._budget_transactions
    
    def get_data(self, use_budget=False, use_aggregated=False):
        if use_budget:
            return self._get_budget()
        elif use_aggregated and self._aggregated_transactions is not None:
            return self._aggregated_transactions
        else:
            return self._get_transactions()
        
    def average_previous_year(self):
        """
        Creates an aggregated version of transactions where the current year's data is replaced
        with past year's category-level averages.
        """
        df = self._get_transactions()
        today = df['date'].max()
        current_year = today.year

        # Define past 12 months range
        one_year_ago = today - pd.DateOffset(years=1)

        # Filter past year transactions
        past_year_data = df[df['date'] >= one_year_ago].copy()

        daily_sums = (
            past_year_data
            .groupby(["date", "csp", 'csp_label'], as_index=False)["amount"]
            .sum()
        )

        previous_year_amounts = (
            daily_sums
            .groupby(['csp'])['amount']
            .sum()
            .to_dict()
        )

        yearly_means = (
            df
            .groupby([df['date'].dt.year, 'csp', 'csp_label'])["amount"]
            .sum()
            .reset_index()
        )
        filt = yearly_means['date'] == current_year
        yearly_means.loc[filt, 'amount'] = (
            yearly_means.loc[filt, 'csp'].map(previous_year_amounts)
        )
        yearly_means['date'] = pd.to_datetime(
            yearly_means['date'].astype(str) + "-01-01"
            ) 
        
        self._aggregated_transactions = yearly_means
        return self._aggregated_transactions

    def filter_and_sum(self, filter_func, use_budget=False, use_aggregated=False,
                       type="custom"):
        """
        Generic method to get past income based on a filter function.
        """
        data = self.get_data(use_budget=use_budget, use_aggregated=use_aggregated)
        filtered_data = filter_func(data)

        if filtered_data.empty:
            print(f'Warning: No data found for {type}')

        agg_data = (
            filtered_data
            .groupby(filtered_data['date'].dt.year)['amount']
            .sum()
        )

        return agg_data

    def scale_current_year(self):
        """Scales current year's transactions to account for incomplete data."""
        today = datetime.date.today()
        current_year = today.year
        days_elapsed = (today - datetime.date(current_year, 1, 1)).days
        scaling_factor = 365 / days_elapsed if days_elapsed > 0 else 1

        data = self._get_transactions()
        if data is not None:
            data.loc[data['date'].dt.year == current_year, 'amount'] *= scaling_factor
        self._transactions = data
        return self._transactions
    
    def project(self, past_data: pd.Series, inflation_factor: float, 
                thru=2100, manual_entries=None):
        """Calculate future fixed income based on past spending trends."""
        previous_year = past_data.index.max()
        previous_amount = past_data[previous_year]

        future_years = range(previous_year + 1, thru + 1)
        projections = pd.Series(index=future_years, dtype='float')

        if manual_entries:
            for year, amount in manual_entries:
                if year in future_years:
                    projections[year] = amount
                else:
                    print(f"⚠️ Warning: Manual entry for {year} is not in projection timeline")

        prev_value = previous_amount
        for year in projections.index:
            if pd.notna(projections[year]):  # Use manual entry if available
                prev_value = projections[year]
            else:
                prev_value *= inflation_factor
                projections[year] = prev_value

        return projections 