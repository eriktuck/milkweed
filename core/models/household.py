import pandas as pd
import numpy as np
from .core import FinancialEntity
from .business import Business
from .individual import Individual
from .transactions import Transactions
import core.utils.functions as functions

class Household(FinancialEntity):
    def __init__(self, name, members: list[Individual], 
                 transactions: Transactions=None, 
                 businesses: list[Business]=[], assets=[]):
        """
        A Household consists of two individuals and their joint expenses, 
        incomes and assets.

        Parameters:
        - name (str): Name to use for filtering transactions
        - members (list[Individual]): List of household members
        - transactions (Transactions, optional): Transactions associated 
        with the household.
        - businesses (list[Income], optional): List of shared household 
        businesses.
        - assets (list[Asset], optional): List of shared assets.
        """
        super().__init__(name, incomes=[], expenses=[])
        self.name = name
        self.members = members
        self.businesses = {business.name: business for business in businesses}
        # self.assets = {asset.name: asset for asset in assets}
        self.transactions = transactions

        # Lazy-loaded attributes
        self._joint_contribution_required = None
        self._taxes = None
        self._allocated_federal_taxes = None

    def add_business(self, business):
        """Add a shared household business source."""
        if not isinstance(business, Business):
            raise TypeError("Only Business instances can be added.")

        if business.name in self.businesses:
            print(f"⚠️ Warning: Business '{business.name}' already exists and will be overwritten.")

        self.businesses[business.name] = business
        self._allocated_federal_taxes = None

    # def add_asset(self, asset):
    #     """Add a shared asset (e.g., house, rental property)."""
    #     if not isinstance(asset, Asset):
    #         raise TypeError("Only Asset instances can be added.")

    #     if asset.name in self.assets:
    #         print(f"⚠️ Warning: Asset '{asset.name}' already exists and will be overwritten.")

    #     self.assets[asset.name] = asset
    
    @property
    def end_year(self):
        return max([member.death_year for member in self.members])
    
    def get_combined_expenses(self):
        """Calculate total household expenses (individual + joint)."""
        # individual_expenses = [person.personal_income.get_personal_expenses() for person in self.members]
        # joint_expenses = [expense.get_stream_series() for expense in self.expenses.values()]
        # business_expenses = [business.get_total_expenses() for business in self.businesses.values()]
        # health_care_expenses = [member.health_care.get_health_costs() for member in self.members]
        # expense_series = individual_expenses + joint_expenses + business_expenses + health_care_expenses

        individual_expenses = []
        for person in self.members:
            if hasattr(person, "personal_income") and person.personal_income is not None:
                individual_expenses.append(person.personal_income.get_personal_expenses())

        # Safely collect health care expenses
        health_care_expenses = []
        for person in self.members:
            if hasattr(person, "health_care") and person.health_care is not None:
                health_care_expenses.append(person.health_care.get_health_costs())

        # Safely collect joint household expenses
        joint_expenses = []
        if self.expenses:
            joint_expenses = [expense.get_stream_series() for expense in self.expenses.values()]

        # Safely collect business expenses
        business_expenses = []
        if self.businesses:
            business_expenses = [business.get_total_expenses() for business in self.businesses.values()]

        # Combine all expenses
        expense_series = individual_expenses + joint_expenses + business_expenses + health_care_expenses

        if expense_series:
            combined_expenses = pd.concat(expense_series, axis=0).groupby(level=0).sum()
            return combined_expenses
        else:
            return pd.Series(dtype=float)


    def get_joint_contribution_required(
            self, tax_function=functions.calculate_married_joint_tax, tol=1
            ):
        """
        Calculate the required gross income to cover expenses & taxes 
        during coast years.

        Parameters:
        - tax_function (function): Function to compute taxes 
        (default: `calculate_married_joint_tax`).
        - tol (float): Tolerance level for convergence.

        Returns:
        - pd.Series: Required gross income per year.
        """
        if self._joint_contribution_required is None:
            # joint_expenses = self.get_total_expenses()
            # business_expenses = [business.get_total_expenses() for business in self.businesses.values()]
            # personal_expenses = [member.personal_income.get_personal_expenses() for member in self.members]
            # health_care_expenses = [member.health_care.get_health_costs() for member in self.members]

            # # Aggregate expenses
            # sum_personal_expenses = pd.concat(personal_expenses).groupby(level=0).sum()
            # sum_business_expenses = (
            #     pd.concat(business_expenses).groupby(level=0).sum() 
            #     if business_expenses else pd.Series(0, index=joint_expenses.index)
            # )
            # sum_health_care_expenses = pd.concat(health_care_expenses).groupby(level=0).sum()
            # sum_expenses = pd.concat([
            #     joint_expenses, sum_personal_expenses, sum_business_expenses, sum_health_care_expenses
            # ]).groupby(level=0).sum()

            sum_expenses = self.get_combined_expenses()

            # Initial estimate of income required (excluding taxes)
            income_required = (-sum_expenses) / 0.7

            def calculate_total_taxes(income: float):
                federal_taxes = tax_function(income)
                fica_taxes = income * (0.062 + 0.0145)
                state_taxes = income * (0.0425 + 0.0045)
                return federal_taxes + fica_taxes + state_taxes

            for _ in range(1000):
                total_taxes = income_required.apply(calculate_total_taxes)
                net_income = income_required - total_taxes
                difference = abs(net_income + sum_expenses)
                if (difference < tol).all():
                    break

                # Adjust guess based on the difference
                adjustment = (-sum_expenses - net_income)
                
                # Use a dampening factor to prevent overshooting
                income_required += adjustment * 0.5

            # Apply business income towards income requirement
            business_incomes = [business.get_total_income() for business in self.businesses.values()]
            total_business_income = (
                pd.concat(business_incomes).groupby(level=0).sum()
                if business_incomes else pd.Series(0, index=income_required.index)
            ).reindex(income_required.index, fill_value=0)
            
            # Deduct personal expenses (to be added back per individual)
            personal_expenses = [member.personal_income.get_personal_expenses() for member in self.members]
            sum_personal_expenses = pd.concat(personal_expenses).groupby(level=0).sum()
            health_care_expenses = [member.health_care.get_health_costs() for member in self.members]
            sum_health_care_expenses = pd.concat(health_care_expenses).groupby(level=0).sum()

            # Calculate full joint contribution required
            joint_contribution_required = -(
                income_required - total_business_income + sum_personal_expenses + sum_health_care_expenses
                )

            self._joint_contribution_required = joint_contribution_required

        return self._joint_contribution_required
    
    def assign_joint_contributions(self):
        """Assign joint contribution evenly between household members."""
        n = len(self.members)
        joint_contribution_required = self.get_joint_contribution_required().div(n)
        for member in self.members:
            member.personal_income.set_joint_contribution_reqd(joint_contribution_required)
    
    def get_combined_adjusted_gross_income(self):
        """
        Return combined gross income for household including personal income 
        (federal wages) and business income (net revenue).
        """
        individual_incomes = [member.personal_income.get_federal_wages() for member in self.members]
        business_incomes = [business.get_net_revenue() for business in self.businesses.values()]

        income_series = individual_incomes + business_incomes
        if income_series:
            combined_incomes = pd.concat(income_series, axis=0).groupby(level=0).sum()
            return combined_incomes
        else:
            return pd.Series(dtype=float)

    def compute_taxes(self, tax_function=functions.calculate_married_joint_tax):
        """
        Compute total household taxes, including federal, Social Security, Medicare, and state taxes.

        Parameters:
        - tax_function (function): Function to compute federal income tax (default: `calculate_married_joint_tax`).

        Returns:
        - pd.Series: Total tax liability per year.
        """
        # Compute taxable wages
        federal_wages = sum(member.personal_income.get_federal_wages() for member in self.members)
        fica_wages = sum(member.personal_income.get_fica_wages() for member in self.members)
        state_wages = sum(member.personal_income.get_state_wages() for member in self.members)

        # Add business income as ordinary income
        business_incomes = [business.get_net_revenue() for business in self.businesses.values()]
        total_business_income = (
            pd.concat(business_incomes, axis=1).sum(axis=1)
            if business_incomes else pd.Series(0, index=federal_wages.index)
        )

        # Combine income for federal tax calculation
        total_taxable_income = pd.concat(
            [federal_wages, total_business_income], axis=0).groupby(level=0).sum()
        federal_taxes = total_taxable_income.apply(tax_function)

        # Compute social security taxes
        individual_fica_wages = [member.personal_income.get_fica_wages() for member in self.members]
        social_security_tax = sum(wage.clip(upper=168600) * 0.062 for wage in individual_fica_wages)

        # Compute Medicare taxes
        medicare_tax = fica_wages * 0.0145
        excess_income = (fica_wages - 250000).clip(lower=0)  # Extra 0.9% for wages over $250,000
        medicare_tax += excess_income * 0.009

        # Compute state taxes
        state_taxes = state_wages * (0.0425 + 0.0045)

        # Compute total taxes
        total_taxes = federal_taxes + social_security_tax + medicare_tax + state_taxes

        # Store for reference
        self._taxes = {
            "federal": federal_taxes,
            "social_security": social_security_tax,
            "medicare": medicare_tax,
            "state": state_taxes,
            "total": total_taxes
        }

        return total_taxes
    
    def allocate_taxes_to_entities(self):
        """
        Computes allocated federal income taxes proportionally by income.
        Social security taxes and medicare taxes can be calculated per
        individual.
        """
        if self._taxes is None:
            self.compute_taxes()

        federal_taxes = self._taxes["federal"]

        # Combine income by entity
        entity_incomes = {}
        for member in self.members:
            name = member.name
            income = member.personal_income.get_federal_wages()
            entity_incomes[name] = income
        
        for business in self.businesses.values():
            name = business.name
            income = business.get_net_revenue()
            entity_incomes[name] = income

        # Align and combine all income series into DataFrame
        income_df = pd.DataFrame(entity_incomes).fillna(0)

        # Total household income by year
        total_income = income_df.sum(axis=1)

        # Avoid divide by zero errors by masking 0-income years
        weights = income_df.div(total_income.replace(0, np.nan), axis=0).fillna(0)
        
        # Allocate taxes per entitiy by multiplying weights
        allocated_federal_taxes = {
            entity: weights[entity] * federal_taxes.reindex(weights.index).fillna(0)
            for entity in weights.columns
        }

        self._allocated_federal_taxes = allocated_federal_taxes
        return allocated_federal_taxes
    
    def get_allocated_federal_taxes(self, entity_name=None):
        """Get allocated federal taxes for an individual or business"""
        if self._allocated_federal_taxes is None:
            self.allocate_taxes_to_entities()
        
        if entity_name:
            return self._allocated_federal_taxes.get(entity_name)
        return self._allocated_federal_taxes
    
    def assign_allocated_taxes(self):
        """Assign taxes proportionally by income."""
        for member in self.members:
            allocated_taxes = self.get_allocated_federal_taxes(entity_name=member.name)
            member.personal_income.set_federal_taxes(allocated_taxes)
        
        for business in self.businesses.values():
            print(business.name)
            allocated_taxes = self.get_allocated_federal_taxes(entity_name=business.name)
            business.set_federal_taxes(allocated_taxes)

    def compute_net_cashflow(self):
        """Compute household net cash flow for contributions and withdrawals."""
        total_income = self.get_combined_adjusted_gross_income()
        total_taxes = self.compute_taxes()
        total_expenses = self.get_combined_expenses()
        return total_income - total_taxes + total_expenses