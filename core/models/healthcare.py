import pandas as pd
import numpy as np

class HealthCare:
    """Models health care costs across working, coast, and retirement years."""

    def __init__(self, individual, employer_premium:float=0, 
                 out_of_pocket: float=0, aca_premium: float=0, 
                 medicare_premium:float=0, end_of_life_cost:float=0):
        """
        Parameters:
        - individual: Reference to the Individual object.
        - employer_premium: Annual employer insurance cost (pre-tax deduction)
        - out_of_pocket: Annual out-of-pocket medical expenses (expense)
        - aca_premium: Annual ACA premium if no employer insurance
        - medicare_premium: Annual Medicare premium
        - end_of_life_cost: Annual expected cost for end-of-life care
        """
        self.individual = individual
        self.employer_premium = employer_premium
        self.out_of_pocket = out_of_pocket
        self.aca_premium = aca_premium
        self.medicare_premium = medicare_premium
        self.end_of_life_cost = end_of_life_cost

    def get_health_costs(self):
        """Returns health care costs based on life stage."""
        years = self.individual.get_scenario_years()
        birth_year = self.individual.birth_year
        ages = np.array(years) - birth_year
        costs = pd.Series(index=years, dtype=float)

        for year, age in zip(years, ages):
            if age < self.individual.coast_age:
                # Working years: Out of pocket (payroll deduction is excluded)
                costs[year] = self.out_of_pocket
            elif year < 65:
                # Coast years: ACA or employer insurance
                costs[year] = self.aca_premium + self.out_of_pocket
            else:
                # Retirement: Medicare + Out-of-pocket
                costs[year] = self.medicare_premium + self.out_of_pocket

        # Add end-of-life care at age 85+
        if self.end_of_life_cost > 0:
            costs.loc[birth_year + 85:] += self.end_of_life_cost

        return -costs

    def get_pre_tax_deductions(self):
        """Returns health insurance deductions (only applies when working)."""
        years = self.individual.get_scenario_years()
        deductions = pd.Series(index=years, dtype=float)

        for year in years:
            if year < self.individual.coast_year:
                # Working years: Employer insurance is a pre-tax deduction
                deductions[year] = self.employer_premium
            else:
                # No pre-tax deduction in coast/retirement years
                deductions[year] = 0

        return deductions
    