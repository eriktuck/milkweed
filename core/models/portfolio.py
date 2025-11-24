import pandas as pd
import numpy as np
import cpi
# cpi.update()  # takes a few minutes!

# Get inflation
years = range(1994, 2025)
cpi_values = {year: cpi.get(year) for year in years}
cpi_series = pd.Series(cpi_values)
INFLATION = cpi_series.pct_change().dropna()

class Holding:
    def __init__(self, symbol, shares, ticker_obj, cost_basis=0.0):
        """
        Parameters
        - symbol (str): like "VTSAX"
        - shares (float): number of shares
        - ticker_obj: Yfinance ticker object
        - cost_basis (float): current cost basis (optional)
        """
        self.symbol = symbol
        self.shares = shares
        self.ticker_obj = ticker_obj

        # Lazy loaded
        self.avg_return = None

        # Internal
        self._price = None

        # Detect if asset is a cash equivalent (based on priceHint)
        try:
            price_hint = ticker_obj.info.get("priceHint", None)
            self.is_cash_equivalent = price_hint == 4
        except Exception as e:
            print(f"Warning: couldn't determine priceHint for {symbol}: {e}")
            self.is_cash_equivalent = False

        # Assign cost basis
        if cost_basis is not None and not pd.isna(cost_basis):
            self.cost_basis = cost_basis
        elif self.is_cash_equivalent:
            self.cost_basis = 1.0 * self.shares
        else:
            self.cost_basis = 0.0

    @property
    def current_price(self):
        """
        Fetches prices from yFinance.

        Returns:
            float: Current price.
        """
        if self._price is None:
            try:
                info = self.ticker_obj.info
                if self.is_cash_equivalent:
                    self._price = 1.0
                elif info.get("priceHint") == 2:
                    self._price = info.get("previousClose")
                else:
                    print(f"[{self.symbol}] Unknown priceHint: {info.get('priceHint')}")
            except Exception as e:
                print(f"[{self.symbol}] Failed to fetch price: {e}")
                self._price = 0.0
        return self._price
    
    @property
    def current_value(self):
        return self.current_price * self.shares
    
    def set_avg_return(self, returns):
        self.avg_return = returns.mean() if isinstance(returns, pd.Series) else returns

    def set_ticker_obj(self, new_ticker):
        self.ticker_obj = new_ticker
        self._price = None
    
    def get_historical_returns(self, period="100y", interval="1mo"):
        """
        Fetches historical annualized returns for each asset.

        Parameters:
        - period (str): Time period (e.g., "10y", "20y").
        - interval (str): The interval for price data (e.g., "1y" for annual returns).

        Returns:
        - float: Historical return.
        """

        hist = (
            self.ticker_obj
            .history(period=period, interval=interval)["Close"]
            .resample("YE").last()
            .pct_change()
            .dropna()
        )
        hist.index = hist.index.year

        return hist
    
    def get_real_returns(self, historical_returns: pd.Series, 
                         inflation_series: pd.Series) -> pd.Series:
        """
        Adjusts historical returns for inflation to compute real returns.

        Parameters:
        - historical_returns (pd.Series): Annualized returns indexed by year.
        - inflation_series (pd.Series): Yearly inflation rates indexed by year.

        Returns:
        - pd.Series: real returns
        """
        common_index = historical_returns.index.intersection(inflation_series.index)
        adjusted_returns = historical_returns.loc[common_index] - inflation_series.loc[common_index]

        return adjusted_returns
    
    def calc_avg_return(self, inflation_series=INFLATION):
        """Calculate inflation adjusted average return."""
        historical_returns = self.get_historical_returns()
        adjusted_returns = self.get_real_returns(historical_returns, inflation_series)
        self.avg_return = adjusted_returns.mean()

    def initialize_forecast_matrix(self, years: pd.Index, 
                                   contributions: pd.Series):
        """
        Create a matrix where each contribution grows forward across the full scenario.

        Parameters:
        - years (pd.Index): all scenario years
        - contributions (pd.Series): contributions aligned with scenario years

        Returns:
        - value_matrix (np.ndarray): future values of contributions
        - cost_matrix (np.ndarray): corresponding cost basis

        Note
        `growth_matrix` is a lower-triangular matrix where each element 
        (i,j) is the growth of the contribution in year j to year i.
        """
        if self.avg_return is None:
            self.calc_avg_return()

        years = years.to_numpy()
        num_years = len(years)
        contribution_array = contributions.to_numpy()

        growth_matrix = (1 + self.avg_return) ** (np.subtract.outer(years, years).clip(min=0))
        growth_matrix = np.tril(growth_matrix)
        
        value_matrix = growth_matrix * contribution_array
        
        cost_matrix = np.zeros_like(value_matrix)
        for i in range(num_years):
            cost_matrix[i:, i] = contribution_array[i]

        # Add initial value
        initial_value = self.current_value
        initial_cost = self.cost_basis
        value_matrix[:, 0] += initial_value * (1 + self.avg_return) ** np.arange(num_years)
        cost_matrix[:, 0] += initial_cost

        return value_matrix, cost_matrix
    
    def forecast_price(self,
                       historical_returns: pd.Series,
                       scenario_years: pd.Index) -> pd.Series:
        """
        Forecasts ticker price over time.

        Parameters:
        - historical_returns (pd.Series): historical returns to forecast
        - scenario_years (pd.Index): years to forecast 

        Returns:
        - pd.Series: forecasted prices per year
        """
        starting_price = self.current_price
        avg_return = historical_returns.mean()
        forecast = starting_price * (1 + avg_return) ** np.arange(len(scenario_years))

        return forecast

class Account:
    def __init__(self, name: str, account_type: str,
                 holdings: list [Holding]=None):
        self.name = name
        self.account_type = account_type
        self.holdings = holdings or []

    def add_holding(self, holding):
        self.holdings.append(holding)
    
    @property
    def current_value(self):
        return sum(holding.current_value for holding in self.holdings)    

class Portfolio:
    def __init__(self, accounts: list[Account]=None):
        self.accounts = accounts or []

    def add_account(self, account):
        self.accounts.append(account)

    @property
    def current_value(self):
        return sum(account.current_value for account in self.accounts)
    
    def bootstrap_portfolio_growth(
            initial_amounts, contribution_plan, 
            historical_returns, simulations=1000):
        """
        Runs a bootstrapping simulation for portfolio growth.

        Parameters:
        - contribution_plan (pd.DataFrame): DataFrame with symbols as columns, years as index.
        - historical_returns (dict): Dictionary of historical returns per asset.
        - simulations (int): Number of simulations.

        Returns:
        - pd.DataFrame: Portfolio value per year after compounding.
        """
        np.random.seed(42)
        scenario_years = contribution_plan.index

        # Prepare results structure with the same MultiIndex columns
        results = np.zeros((len(scenario_years), simulations, len(contribution_plan.columns)))
        
        for col_idx, (account, symbol) in enumerate(contribution_plan.columns):
            if symbol not in historical_returns:
                # raise ValueError(f"Missing historical returns for {symbol}")
                returns = historical_returns['VTSAX']

            returns = historical_returns[symbol]

            for sim in range(simulations):
                portfolio_value = initial_amounts.loc[(account, symbol)]
                for i, year in enumerate(scenario_years):
                    # Apply contribution for this account-symbol pair
                    contribution_value = contribution_plan.loc[year, (account, symbol)]
                    portfolio_value += contribution_value

                    # Randomly select a return from history
                    random_return = np.random.choice(returns)
                    portfolio_value *= (1 + random_return)

                    # Store result
                    results[i, sim, col_idx] = portfolio_value

        return results
    