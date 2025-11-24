import pandas as pd
from .transactions import Transactions

class Stream:
    """
    A general class to get historic and projected streams of
    transactions. Expects historic expenses to be negative, incomes
    to be positive.
    """

    def __init__(self, name, transactions: Transactions, filter_func, 
                 end_year=2100, inflation_factor=1.0, use_budget=False):
        """
        Parameters:
        - name (str): Name of the income source.
        - transactions (Transactions): Reference to transactions object.
        - filter_func (callable): Function to filter relevant income from transactions.
        - inflation_factor (float): Inflation factor for projection.
        - use_budget (bool): Whether to use budget data instead of past transactions.
        """
        self.name = name
        self.transactions = transactions
        self.filter_func = filter_func
        self.end_year = end_year
        self.inflation_factor = inflation_factor
        self.use_budget = use_budget
        self._past_stream = None
        self._projected_stream = None
        self._combined_stream = None
        self._manual_entries = []

    def add_manual_entry(self, year, amount):
        """Manually add expected income for a specific year."""
        self._manual_entries.append((year, amount))
        self._projected_income = None  # Invalidate cache

    def get_past_stream(self):
        """Retrieve past stream from transactions."""
        if self._past_stream is None:
            # Modify past transactions if not using the budget
            if not self.use_budget:
                self.transactions.average_previous_year()
                use_aggregated = True
            else:
                use_aggregated = False

            # Filter and aggregate transactions
            self._past_stream = self.transactions.filter_and_sum(
                self.filter_func, use_budget=self.use_budget,
                use_aggregated=use_aggregated, type=self.name
            )

        return self._past_stream

    def get_projected_stream(self):
        """Project future stream based on past trends or budget."""
        if self._projected_stream is None:
            past_data = self.get_past_stream()

            # If no past data in budget, use transactions 
            if self.use_budget and past_data.empty:
                print(f"⚠️ Warning: No budget data found for {self.name}. Falling back on scaled transactions.")
                past_data = self.transactions.average_previous_year()

            self._projected_stream = self.transactions.project(
                past_data,
                self.inflation_factor,
                self.end_year,
                self._manual_entries
            )
        return self._projected_stream

    def get_stream_series(self):
        """Retrieve total expenses (past + projected)."""
        if self._combined_stream is None:
            self._combined_stream = pd.concat([
                self.get_past_stream(),
                self.get_projected_stream()
            ]).sort_index()
        return self._combined_stream
    
class FinancialEntity:
    """A base class for any entity that has income and expenses."""

    def __init__(self, name, incomes: list[Stream]=None, 
                 expenses: list[Stream]=None):
        """
        Parameters:
        - name (str): Name of the entity.
        - incomes (list[Stream]): List of income sources.
        - expenses (list[Stream]): List of expense sources.
        """
        self.name = name
        self.incomes = {income.name: income for income in (incomes or [])}
        self.expenses = {expense.name: expense for expense in (expenses or [])}

    def add_income(self, income):
        """Add an income stream to the entity."""
        if not isinstance(income, Stream):
            raise TypeError("Only instances of Stream can be added.")
        self.incomes[income.name] = income

    def add_expense(self, expense):
        """Add an expense to the entity."""
        if not isinstance(expense, Stream):
            raise TypeError("Only instances of Stream can be added.")
        self.expenses[expense.name] = expense

    def get_total_income(self):
        """Aggregate all income streams into a single total."""
        income_series = [income.get_stream_series() for income in self.incomes.values()]
        if not income_series:
            return pd.Series(dtype=float)
        return pd.concat(income_series, axis=1).sum(axis=1)

    def get_total_expenses(self):
        """Aggregate all expenses into a single total."""
        expense_series = [expense.get_stream_series() for expense in self.expenses.values()]
        if not expense_series:
            return pd.Series(dtype=float)
        return pd.concat(expense_series, axis=1).sum(axis=1)
    
    def get_net_cashflow(self):
        """Calculates net cashflow"""
        return self.get_total_income() + self.get_total_expenses()
    

