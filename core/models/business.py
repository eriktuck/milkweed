import pandas as pd
from .core import FinancialEntity, Stream
from .transactions import Transactions

class Business(FinancialEntity):
    """Represents a business that generates revenue and has expenses."""

    def __init__(self, name, ownership, transactions:Transactions=None,
                 exit_year: int=2100, write_offs: list[Stream]=None):
        """
        Parameters:
        - name (str): Business name.
        - ownership (dict): Defines owner percentages.
        - transaction (Transaction): Transaction table.
        - write_offs (list[Stream]): List of write_off sources, prorated if necessary.
        """
        super().__init__(name, incomes=[], expenses=[])
        self.ownership = ownership
        self.transactions = transactions
        self.exit_year = exit_year
        self.write_offs = {write_off.name: write_off for write_off in (write_offs or [])}
        self._assigned_federal_taxes = None # Assigned by household
        
    def add_write_off(self, write_off):
        """Add a write_off to the entity."""
        if not isinstance(write_off, Stream):
            raise TypeError("Only instances of Stream can be added.")
        self.write_offs[write_off.name] = write_off

    def get_net_revenue(self):
        cashflow = self.get_net_cashflow()
        write_offs = [write_off.get_stream_series() for write_off in self.write_offs.values()]
        net_revenue = [cashflow] + write_offs
        if not net_revenue:
            return pd.Series(dtype=float)
        return pd.concat(net_revenue, axis=1).sum(axis=1)

    def get_income_distribution(self):
        """Distribute net income based on ownership percentages."""
        net_income = self.get_net_cashflow()
        return {owner: net_income * percent for owner, percent in self.ownership.items()}
    
    def set_federal_taxes(self, federal_taxes):
        self._assigned_federal_taxes = federal_taxes
    
    def get_assigned_taxes(self):
        return self._assigned_federal_taxes
    
    def calculate_excess_pay(self):
        excess_pay = self.get_net_cashflow() + self.get_assigned_taxes()
        return excess_pay
    