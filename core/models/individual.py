import pandas as pd
from .core import Stream, FinancialEntity
from .healthcare import HealthCare
from .business import Business
import core.utils.functions as functions

class Individual(FinancialEntity):
    """
    Represents an individual with incomes, expenses, and financial 
    planning attributes.
    """

    def __init__(self, name, birth_year,  
                 coast_age=50, retirement_age=67, 
                 death_age=90, claim_age=70, transactions=None):
        """
        Initialize an Individual with financial attributes.

        Parameters:
        - name (str): Name of the individual.
        - birth_year (int): Year of birth.
        - coast_age (int): Age when Coast FIRE starts.
        - retirement_age (int): Age when retirement starts.
        - death_age (int): Expected age at death.
        - claim_age (int): Age at which Social Security is claimed.
        - transactions (Transactions): The Transactions instance for this person.
        """
        super().__init__(name)

        self.birth_year = birth_year
        self._death_age = death_age
        self._coast_age = coast_age
        self._retirement_age = retirement_age
        self._claim_age = claim_age
        self.transactions = transactions

        # Personal Income
        self.personal_income = PersonalIncome(self)
        self.add_income(self.personal_income)
        
        # Health Care
        self.health_care = None

        # Lazy-loaded attributes
        self._working_years = None
        self._coast_years = None
        self._retirement_years = None
        self._scenario_years = None

    @property
    def death_age(self):
        return self._death_age
    
    @death_age.setter
    def death_age(self, new_death_age):
        self._death_age = new_death_age
        self._retirement_years = None  # Reset cached values

    @property
    def death_year(self):
        return self.birth_year + self._death_age

    @property
    def coast_age(self):
        return self._coast_age
    
    @coast_age.setter
    def coast_age(self, new_coast_age):
        if new_coast_age > self._retirement_age:
            raise ValueError("Coast age must be less than retirement age.")
        self._coast_age = new_coast_age
        self._working_years = None
        self._coast_years = None

    @property
    def retirement_age(self):
        return self._retirement_age
    
    @retirement_age.setter
    def retirement_age(self, new_retirement_age):
        if new_retirement_age < self._coast_age:
            raise ValueError("Retirement age must be greater than or equal to coast age.")
        self._retirement_age = new_retirement_age
        self._coast_years = None
        self._retirement_years = None
    
    @property
    def claim_age(self):
        return self._claim_age
    
    @claim_age.setter
    def claim_age(self, new_claim_age):
        if new_claim_age < 62 or new_claim_age > 70:
            raise ValueError("Claim age must be between 62 and 70.")
        self._claim_age = new_claim_age

    @property
    def coast_year(self):
        return self.birth_year + self.coast_age

    @property
    def retirement_year(self):
        return self.birth_year + self.retirement_age

    @property
    def claim_year(self):
        return self.birth_year + self.claim_age

    def get_working_years(self) -> range:
        if self._working_years is None:
            self._working_years = range(2025, self.coast_year)
        return self._working_years
    
    def get_coast_years(self) -> range:
        if self._coast_years is None:
            self._coast_years = range(self.coast_year, self.retirement_year)
        return self._coast_years
    
    def get_retirement_years(self) -> range:
        if self._retirement_years is None:
            self._retirement_years = range(self.retirement_year, self.death_year + 1)
        return self._retirement_years
    
    def get_scenario_years(self) -> range:
        if self._scenario_years is None:
            self._scenario_years = range(2025, self.death_year + 1)
        return self._scenario_years
    
    def assign_healthcare(self, health_care: HealthCare):
        if not isinstance(health_care, HealthCare):
            raise TypeError("Only instances of HealthCare can be assigned.")
        self.health_care = health_care

    def add_business(self, business: Business):
        """Dynamically add a new Business instance to the individual."""
        if not isinstance(business, Business):
            raise TypeError("Only instances of Business can be added.")
        self.add_income(business)  # Businesses contribute to income

class PersonalIncome(Stream):
    """
    Represents personal income for an individual, including salaries and social security.
    Handles:
    - User-supplied past incomes
    - Manual adjustments for future earnings
    - Inflation-based projections
    """

    def __init__(self, individual: Individual, inflation_factor=1.03):
        """
        Initialize a PersonalIncome instance.

        Parameters:
        - individual (Individual): Reference to the owning individual.
        - inflation_factor (float): Default inflation factor for projecting income.
        """
        super().__init__(name="Paychecks", 
                         transactions=individual.transactions, 
                         filter_func=lambda df: df[(df['csp'] == 'income')],
                         inflation_factor=inflation_factor)
        self.individual = individual
        self._past_gross_income = None  # User-supplied past earnings
        self._manual_income_updates = []  # List of (year, amount) manual entries
        self._gross_income_to_coast = None
        self._gross_income_in_coast = None
        self._social_security_benefits = None
        self._employer_match = 0.03

        self.pre_tax_contributions = []  # List of PreTaxContribution objects
        self._joint_contribution_reqd = None
        self._assigned_federal_taxes = None # Assigned by household    
        self.portfolio = None  # Portfolio object

    @property
    def past_gross_income(self):
        return self._past_gross_income

    @past_gross_income.setter
    def past_gross_income(self, value):
        if not isinstance(value, pd.Series):
            raise TypeError("past_gross_income must be a Pandas Series")
        if not isinstance(value.index, pd.Index) or not pd.api.types.is_integer_dtype(value.index):
            raise ValueError("past_gross_income index must be years as integers (e.g., 2010, 2011).")
        self._past_gross_income = value.sort_index()  # Ensure chronological order

    def _validate_past_income(self):
        """Ensures past_gross_income is set before using it."""
        if self._past_gross_income is None:
            raise ValueError("Please set `past_gross_income` as a Pandas Series with years as the index.")

    def add_manual_income_entry(self, year, amount):
        """Allow user to manually set expected income for specific years."""
        self._manual_income_updates.append((year, amount))
        self._gross_income_to_coast = None  # Invalidate cache to recompute

    def get_personal_expenses(self, joint_id='Joint Contribution'):
        """Returns total expenses less joint contribution. Excludes healthcare."""
        joint_contribution = self.individual.expenses[joint_id].get_stream_series()
        return self.individual.get_total_expenses() - joint_contribution
        
    def set_joint_contribution_reqd(self, joint_contribution):
        self._joint_contribution_reqd = joint_contribution
        self._gross_income_in_coast = None

    def get_gross_income_to_coast(self, inflation_factor=None):
        """Project future income until Coast FIRE year."""
        self._validate_past_income()
        
        # Use stored inflation factor if none provided
        if inflation_factor is None:
            inflation_factor = self.inflation_factor
        
        # If not computed or inflation factor changed, recompute
        if (self._gross_income_to_coast is None or
            self.inflation_factor != inflation_factor):
            
            self._gross_income_to_coast = self.transactions.project(
                self._past_gross_income, 
                inflation_factor, 
                self.individual.death_year,
                self._manual_income_updates
            ).loc[self.individual.get_working_years()]

            self.inflation_factor = inflation_factor

        return self._gross_income_to_coast

    def get_gross_income_in_coast(self):
        """
        Assign required joint contribution to coast years.
        """
        if self._gross_income_in_coast is None:
            personal_expenses = self.get_personal_expenses()
            health_care_expenses = self.individual.health_care.get_health_costs()
            if self._joint_contribution_reqd is None:
                print("⚠️ Warning: Joint contribution not set. Gross income will be equal to personal expenses")
            joint_contribution = self._joint_contribution_reqd
            income_required = pd.concat([personal_expenses, health_care_expenses, joint_contribution]).groupby(level=0).sum()
            self._gross_income_in_coast = (
                -(income_required.loc[self.individual.get_coast_years()])
            )
        return self._gross_income_in_coast

    def get_gross_income(self, inflation_factor=None):
        """Aggregate past and projected income, including manual adjustments."""
        self._validate_past_income()

        return pd.concat([
            self.past_gross_income,
            self.get_gross_income_to_coast(inflation_factor),
            self.get_gross_income_in_coast()
        ], axis=0)

    def add_pre_tax_contribution(self, contribution):
        """Adds a pre-tax contribution to be deducted from gross income."""
        self.pre_tax_contributions.append(contribution)

    def calculate_pre_tax_deductions(self):
        """
        Calculates total pre-tax deductions by summing contributions 
        from all added PreTaxContribution objects. Ensures that 
        overlapping contributions are summed while handling 
        non-overlapping years correctly. Also deducts healthcare.

        Returns:
        - pd.Series: A series with total pre-tax deductions for each year.
        """
        # Get gross income
        gross_income = self.get_gross_income()
        
        # Get all pre-tax contributions (401k, HSA)
        pre_tax_contributions = sum(
            contrib.calculate_contribution(gross_income).reindex(gross_income.index, fill_value=0)
            for contrib in self.pre_tax_contributions
        ) if self.pre_tax_contributions else pd.Series(0, index=gross_income.index)

        # Add employer-sponsored health insurance
        health_deductions = self.individual.health_care.get_pre_tax_deductions()

        return pre_tax_contributions + health_deductions
    
    def calculate_hsa_deductions(self):
        # Get gross income
        gross_income = self.get_gross_income()
        
        # Sum contributions while ensuring non-overlapping series are aligned properly
        if not self.pre_tax_contributions:
            return pd.Series(0, index=gross_income.index)

        total_deductions = sum(
            contrib.calculate_contribution(gross_income).reindex(gross_income.index, fill_value=0)
            for contrib in self.pre_tax_contributions if contrib.name == "HSA"
        )
        
        return total_deductions

    def get_federal_wages(self):
        """Returns taxable wages for federal income tax."""
        gross_income = self.get_gross_income()
        pre_tax_deductions = self.calculate_pre_tax_deductions()
        pre_tax_deductions = pre_tax_deductions.reindex(gross_income.index, fill_value=0)
        return gross_income - pre_tax_deductions

    def get_fica_wages(self):
        """Returns wages subject to Social Security and Medicare taxes."""
        gross_income = self.get_gross_income()
        hsa_deductions = self.calculate_hsa_deductions()  # FICA does not deduct 401k
        hsa_deductions = hsa_deductions.reindex(gross_income.index, fill_value=0)
        return gross_income - hsa_deductions
    
    def get_medicare_wages(self):
        gross_income = self.get_gross_income()
        hsa_deductions = self.calculate_hsa_deductions() # FICA does not deduct 401k
        hsa_deductions = hsa_deductions.reindex(gross_income.index, fill_value=0)
        return gross_income - hsa_deductions

    def get_state_wages(self):
        """Returns taxable wages for state income tax."""
        gross_income = self.get_gross_income()
        pre_tax_deductions = self.calculate_pre_tax_deductions()  # CO allows HSA and 401k deductions
        pre_tax_deductions = pre_tax_deductions.reindex(gross_income.index, fill_value=0)
        return gross_income - pre_tax_deductions
    
    def set_federal_taxes(self, federal_taxes):
        self._assigned_federal_taxes = federal_taxes

    def get_assigned_taxes(self):
        return self._assigned_federal_taxes

    def calculate_net_pay(self):
        """Calculates net pay after social security, medicare, and pre-tax deductions."""
        taxable_income = self.get_federal_wages()
        
        # Get assigned federal taxes
        assigned_federal_taxes = self.get_assigned_taxes()
        
        # Get social security taxes
        fica_wage = self.get_fica_wages()
        social_security_tax = fica_wage.clip(upper=168600) * 0.062

        # Compute Medicare taxes
        medicare_tax = fica_wage * 0.0145
        excess_income = (fica_wage - 250000).clip(lower=0)  # Extra 0.9% for wages over $250,000
        medicare_tax += excess_income * 0.009

        # Compute state taxes
        state_taxes = self.get_state_wages() * (0.0425 + 0.0045)

        # Compute total taxes
        total_taxes = assigned_federal_taxes + social_security_tax + medicare_tax + state_taxes

        # Net pay after payroll taxes
        net_pay = taxable_income - total_taxes

        return net_pay
    
    def calculate_excess_pay(self, joint_contribution):
        """
        Return the amount available for contributions.
        
        Parameters
        - joint_contribution (pd.Series): Half of household expenses
        """
        net_pay = self.calculate_net_pay()
        personal_expenses = self.get_personal_expenses()
        healthcare_expenses = self.individual.health_care.get_health_costs()

        excess_pay = net_pay + personal_expenses + healthcare_expenses + joint_contribution

        return excess_pay
        
    def get_social_security_benefits(self):
        if self._social_security_benefits is None:
            ss_years = range(self.individual.claim_year, self.individual.death_year + 1)
            hsa_contribution = self.calculate_hsa_deductions()
            ss_income = self.get_gross_income() - hsa_contribution
            ss_benefit = functions.calculate_social_security_benefit(
                ss_income, 
                self.individual.claim_age
                )
            self._social_security_benefits = pd.Series(ss_benefit * 12, index=ss_years)
        return self._social_security_benefits
    

class PreTaxContribution:
    """
    Represents a pre-tax contribution such as a 401(k) or HSA, allowing for 
    user-defined start and end years with year-specific maximum contribution limits.
    """
    
    def __init__(self, name, rate, max_contribution: pd.Series, 
                 start_year=None, end_year=None, matched=True):
        """
        Parameters:
        - name (str): The name of the contribution (e.g., "401k", "HSA").
        - rate (float): The percentage of gross income to contribute.
        - max_contribution (pd.Series): A Pandas Series with years as index and max contribution limits as values.
        - start_year (int, optional): First year to apply contributions.
        - end_year (int, optional): Last year to apply contributions.
        - matched (bool, optional): Whether contribution is employer matched
        """
        self.name = name
        self.rate = rate
        self.max_contribution = max_contribution
        self.start_year = start_year
        self.end_year = end_year
        self.matched = matched

    def calculate_contribution(self, gross_income: pd.Series) -> pd.Series:
        """
        Calculate the pre-tax contribution based on gross income.

        Parameters:
        - gross_income (pd.Series): A Pandas Series with years as index and income as values.

        Returns:
        - pd.Series: The calculated pre-tax contributions for each year.
        """
        # Ensure that the gross_income and max_contribution indices align
        applicable_years = gross_income.index.intersection(self.max_contribution.index)

        # Filter income and max contribution limits to valid years
        filtered_income = gross_income.loc[applicable_years]
        max_limits = self.max_contribution.loc[applicable_years]

        # Apply start and end year limits
        if self.start_year:
            filtered_income = filtered_income[filtered_income.index >= self.start_year]
        if self.end_year:
            filtered_income = filtered_income[filtered_income.index <= self.end_year]

        # Ensure max_limits matches the filtered income index
        max_limits = max_limits.reindex(filtered_income.index, fill_value=max_limits.max())

        # Calculate contributions as a percentage of gross income
        contributions = filtered_income * self.rate

        # Clip contributions to the max allowable limit per year
        contributions = contributions.clip(upper=max_limits)

        return contributions

