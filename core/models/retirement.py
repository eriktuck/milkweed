import pandas as pd
import numpy as np
from .portfolio import Holding, Account, Portfolio

class RetirementScenario:
    def __init__(self, portfolio: Portfolio, start_year: int, 
                 end_year: int, start_age: int, expenses: pd.Series, 
                 contributions: dict):
        self.portfolio = portfolio
        self.years = pd.Index(range(start_year, end_year + 1))
        self.start_age = start_age
        self.expenses = expenses
        self.contributions = contributions  # {(account_name, symbol): pd.Series}
        self.forecasts = {}  # {(account, symbol): {'value': ..., 'cost': ..., 'withdrawals': ...}}

    def initialize(self):
        for account in self.portfolio.accounts:
            for holding in account.holdings:
                key = (account.name, holding.symbol)
                if key not in self.contributions:
                    raise ValueError(f"Missing contributions for {holding.symbol} in {account.name}")
                
                # Cast contributions across scenario years and fill 0s
                full_contributions = pd.Series(0.0, index=self.years)
                full_contributions.update(self.contributions[key])
                
                # Initialize value matrix and cost basis matrix
                value_matrix, cost_matrix = holding.initialize_forecast_matrix(
                    years=self.years,
                    contributions=full_contributions
                )

                # Initialize forecasts
                self.forecasts[key] = {
                    'value': value_matrix,
                    'cost': cost_matrix,
                    'withdrawals': np.zeros(len(self.years))
                }

    def withdraw_from_holding(self, holding, account_name, symbol, year_idx, amount):
        forecast = self.forecasts[(account_name, symbol)]
        value_matrix = forecast['value']
        cost_matrix = forecast['cost']
        withdrawals = forecast['withdrawals']

        value_at_year = np.sum(value_matrix[year_idx])
        cost_at_year = np.sum(cost_matrix[year_idx])

        if value_at_year <= 0:
            return None

        amount = min(amount, value_at_year)
        gain_ratio = 1 - (cost_at_year / value_at_year)
        capital_gains = round(amount * gain_ratio, 2)
        cost_basis_used = round(amount - capital_gains, 2)

        proportions = value_matrix[year_idx] / value_at_year
        proportions = np.nan_to_num(proportions)
        reduction = proportions * amount

        for i in range(year_idx, len(self.years)):
            growth = (1 + holding.avg_return) ** (i - year_idx)
            value_matrix[i] -= reduction * growth
            cost_matrix[i] -= proportions * amount
            value_matrix[i] = np.clip(value_matrix[i], 0.0, None)
            cost_matrix[i] = np.clip(cost_matrix[i], 0.0, value_matrix[i])

        withdrawals[year_idx] += amount

        return {
            'symbol': symbol,
            'year': self.years[year_idx],
            'amount': round(amount, 2),
            'cost_basis_used': cost_basis_used,
            'capital_gains': capital_gains
        }

    def withdraw_for_year(self, year_idx, year, age, amount_needed):
        remaining = amount_needed
        withdrawals = []
        taxable_income = 0.0
        capital_gains = 0.0

        priorities = ['taxable', 'trad_ira', 'roth_ira']
        for acct_type in priorities:
            if acct_type in ['trad_ira', 'roth_ira'] and age < 59.5:
                continue

            for account in self.portfolio.accounts:
                if account.account_type != acct_type:
                    continue

                for holding in account.holdings:
                    key = (account.name, holding.symbol)
                    value_matrix = self.forecasts[key]['value']
                    if np.sum(value_matrix[year_idx]) <= 0:
                        continue

                    amount_to_withdraw = min(remaining, np.sum(value_matrix[year_idx]))
                    result = self.withdraw_from_holding(
                        holding, account.name, holding.symbol, year_idx, amount_to_withdraw
                    )
                    if result:
                        result.update({'account': account.name, 'account_type': acct_type})
                        withdrawals.append(result)
                        remaining -= result['amount']
                        if acct_type == 'taxable':
                            capital_gains += result['capital_gains']
                        elif acct_type == 'trad_ira':
                            taxable_income += result['amount']
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        return {
            'withdrawals': withdrawals,
            'total_withdrawn': round(amount_needed - remaining, 2),
            'taxable_income': round(taxable_income, 2),
            'capital_gains': round(capital_gains, 2),
            'remaining': round(remaining, 2)
        }

    def simulate(self):
        self.initialize()
        records = []

        for i, year in enumerate(self.years):
            age_in_year = self.start_age + i
            amount_needed = -self.expenses.get(year, 0.0)

            result = self.withdraw_for_year(i, year, age_in_year, amount_needed)
            result.update({'year': year, 'age': age_in_year})
            records.append(result)

        return pd.DataFrame(records).set_index('year')

    def forecast_total_value(self):
        total_by_year = {year: 0.0 for year in self.years}
        for forecast in self.forecasts.values():
            yearly_values = np.sum(forecast['value'], axis=1)
            for i, year in enumerate(self.years):
                total_by_year[year] += yearly_values[i]
        return total_by_year

    def summary(self):
        return {
            "starting_value": self.portfolio.current_value,
            "years": (self.years[0], self.years[-1]),
            "accounts": len(self.portfolio.accounts),
            "start_age": self.start_age
        }
