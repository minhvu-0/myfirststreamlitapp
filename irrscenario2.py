import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import norm, uniform, beta
from typing import List, Dict, Tuple, Any

# Distribution generators
class Distributions:
    @staticmethod
    def pert(min_val, mode_val, max_val, size=1):
        alpha, beta_range = 4.0, max_val - min_val
        alpha1 = 1 + alpha * ((mode_val - min_val) / beta_range)
        alpha2 = 1 + alpha * ((max_val - mode_val) / beta_range)
        return min_val + np.random.beta(alpha1, alpha2, size=size) * beta_range
    
    @staticmethod
    def triangular(min_val, mode_val, max_val, size=1):
        c = (mode_val - min_val) / (max_val - min_val)
        return stats.triang.rvs(c, loc=min_val, scale=(max_val - min_val), size=size)
    
    @staticmethod
    def uniform(min_val, max_val, size=1):
        return np.random.uniform(min_val, max_val, size=size)
    
    @staticmethod
    def normal(mean, std_dev, size=1):
        return np.random.normal(mean, std_dev, size=size)
    
    @staticmethod
    def lognormal(mean, percentile_95, size=1):
        sigma = np.log(percentile_95 / mean) / 1.645
        mu = np.log(mean) - (sigma**2) / 2
        return np.random.lognormal(mu, sigma, size=size)

# Investment simulation class
class InvestmentSimulator:
    def __init__(self, cash_flows, current_loan, dist_types, dist_params,
                 base_year, dev_time, debt_pct=0.65, financing_rate=0.10,
                 immediate_costs_pct=0.026, future_costs_pct=0.018):
        self.cash_flows = cash_flows
        self.current_loan = current_loan  # Current loan amount that must be repaid at sale
        self.dist_types = dist_types
        self.dist_params = dist_params
        self.base_year = base_year
        self.dev_time = dev_time
        self.debt_pct = debt_pct  # Debt as percentage of total development costs
        self.financing_rate = financing_rate  # Annual financing rate
        self.immediate_costs_pct = immediate_costs_pct
        self.future_costs_pct = future_costs_pct
    
    def _generate_value(self, param_name, size=1):
        dist_type = self.dist_types[param_name]
        params = self.dist_params[param_name]
        
        if dist_type == 'PERT':
            return Distributions.pert(params['min'], params['mode'], params['max'], size)
        elif dist_type == 'Triangular':
            return Distributions.triangular(params['min'], params['mode'], params['max'], size)
        elif dist_type == 'Uniform':
            return Distributions.uniform(params['min'], params['max'], size)
        elif dist_type == 'Normal':
            return Distributions.normal(params['mean'], params['std'], size)
        elif dist_type == 'LogNormal':
            return Distributions.lognormal(params['mean'], params['percentile_95'], size)
    
    @staticmethod
    def calculate_irr(cash_flows):
        try:
            return npf.irr(cash_flows) * 100
        except:
            return np.nan
    
    def simulate_immediate_exit(self, num_sims):
        irrs = np.zeros(num_sims)
        exit_prices = np.zeros(num_sims)
        selling_costs_values = np.zeros(num_sims)
        net_proceeds_values = np.zeros(num_sims)
        profit_on_cost_values = np.zeros(num_sims)
        
        for i in range(num_sims):
            exit_price = self._generate_value('immediate_exit_price', size=1)[0]
            exit_prices[i] = exit_price
            
            selling_costs = exit_price * self.immediate_costs_pct
            selling_costs_values[i] = selling_costs
            
            # Must repay the current loan when selling
            net_proceeds = exit_price - selling_costs - self.current_loan
            net_proceeds_values[i] = net_proceeds

            # Define profit on cost
            total_costs = sum(abs(cf) for cf in self.cash_flows if cf < 0) + self.current_loan
            profit_on_cost = (net_proceeds / total_costs) * 100 if total_costs > 0 else 0
            profit_on_cost_values[i] = profit_on_cost
            
            # Calculate IRR
            cf = self.cash_flows.copy()
            cf.append(net_proceeds)
            irrs[i] = self.calculate_irr(cf)
        
        return pd.DataFrame({
            'IRR': irrs,
            'Exit_Price': exit_prices,
            'Selling_Costs': selling_costs_values,
            'Net_Proceeds': net_proceeds_values,
            'Profit_on_Cost': profit_on_cost_values 
        })
    
    def simulate_development(self, num_sims):
        irrs = np.zeros(num_sims)
        future_prices = np.zeros(num_sims)
        construction_costs = np.zeros(num_sims)
        delays = np.zeros(num_sims)
        additional_equity_required = np.zeros(num_sims)
        debt_amounts = np.zeros(num_sims)
        financing_costs_total = np.zeros(num_sims)
        total_development_costs = np.zeros(num_sims)
        exit_years = np.zeros(num_sims, dtype=int)
        land_prices = np.zeros(num_sims)
        profit_on_cost_values = np.zeros(num_sims) #new
        
        # First, simulate immediate exit to get land prices
        immediate_exit_df = self.simulate_immediate_exit(num_sims)
        
        for i in range(num_sims):
            # Get the land price from immediate exit price
            land_price = immediate_exit_df['Exit_Price'][i]
            land_prices[i] = land_price
                        
            # Current loan to be refinanced
            current_loan = self.current_loan

            # Generate parameters
            future_price = self._generate_value('future_exit_price', size=1)[0]
            future_prices[i] = future_price
            
            construction_cost = self._generate_value('development_cost', size=1)[0]
            construction_costs[i] = construction_cost
            
            delay = self._generate_value('additional_delay', size=1)[0]
            if 'cap' in self.dist_params['additional_delay']:
                delay = min(delay, self.dist_params['additional_delay']['cap'])
            delays[i] = delay
            
            # Calculate exit year
            exit_year = int(round(self.base_year + self.dev_time + delay))
            exit_years[i] = exit_year
            
            # Calculate development time in years (for financing cost calculation)
            development_time = self.dev_time + delay
            
            # Calculate total development costs
            total_dev_cost = land_price + construction_cost
            
            # Calculate financing costs using the financing rate parameter
            financing_cost = total_dev_cost * self.financing_rate * development_time
            financing_costs_total[i] = financing_cost
            
            # Update total development costs to include financing
            total_dev_cost += financing_cost
            total_development_costs[i] = total_dev_cost
            
            # Calculate funding sources
            soft_equity = land_price - current_loan  # Land price is considered soft equity
            debt = total_dev_cost * self.debt_pct  # Debt percentage from parameters
            debt_amounts[i] = debt
            
            # Additional equity needed
            add_equity = total_dev_cost - debt - soft_equity
            additional_equity_required[i] = max(0, add_equity)  # Ensure no negative equity
            
            # Create cash flow array
            cf = self.cash_flows.copy()
            
            # Add land acquisition (soft equity) at base year
            # Note: Since we're assuming the land is already owned, we just account for
            # the additional equity needed for construction and financing
            while len(cf) <= self.base_year:
                cf.append(0)
            cf[self.base_year] = -max(0, add_equity)
            
            # Add zeros for interim years
            while len(cf) < exit_year:
                cf.append(0)
            
            # Add exit proceeds
            selling_costs = future_price * self.future_costs_pct
            net_proceeds = future_price - selling_costs - debt
            cf.append(net_proceeds)

            # Calculate Profit on Cost
            total_dev_cost = total_development_costs[i]  # Already calculated earlier in the loop
            profit_on_cost = (net_proceeds / total_dev_cost) * 100 if total_dev_cost > 0 else 0
            profit_on_cost_values[i] = profit_on_cost
            
            # Calculate IRR
            irrs[i] = self.calculate_irr(cf)
        
        return pd.DataFrame({
            'IRR': irrs,
            'Future_Exit_Price': future_prices,
            'Land_Price': land_prices,
            'Current_Loan': self.current_loan,
            'Construction_Cost': construction_costs,
            'Total_Development_Cost': total_development_costs,
            'Financing_Cost': financing_costs_total,
            'Debt': debt_amounts,
            'Soft_Equity': land_prices,  # Now this is the same as land price
            'Additional_Equity': additional_equity_required,
            'Additional_Delay': delays,
            'Exit_Year': exit_years,
            'Profit_on_Cost': profit_on_cost_values
        })
    
    def run_simulation(self, num_sims=10000):
        return (
            self.simulate_immediate_exit(num_sims),
            self.simulate_development(num_sims)
        )

# Dashboard function
def run_dashboard():
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    st.title("Real Estate Investment Analysis")
    
    # Settings sidebar
    st.sidebar.header("Simulation Settings")
    num_sims = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000, 1000)
    random_seed = st.sidebar.number_input("Random Seed", 0, 100000, 42)
    np.random.seed(random_seed)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Inputs", "Distribution Settings", "Results"])
    
    # Tab 1: Inputs
    with tab1:
        st.header("Investment Parameters")
        
        # Historical cash flows
        st.subheader("Historical Cash Flows")
        cash_flow_data = pd.DataFrame({
            'Year': [2014, 2019, 2020, 2021, 2022, 2023, 2024],
            'Amount': [-1800000, -2600000, -600000, 2100000, 300000, -640000, -500000]
        })
        
        edited_cf = st.data_editor(
            cash_flow_data,
            hide_index=True,
            key="cash_flows"
        )
        
        base_cash_flows = edited_cf['Amount'].tolist()
        
        # Loan and other parameters
        col1, col2 = st.columns(2)
        with col1:
            current_loan = st.number_input("Current Loan (£)", 0, 50000000, 13700000, 100000)
            standard_dev_time = st.slider("Base Development Time (years)", 1, 7, 3)
            debt_pct = st.slider("Debt Percentage (%)", 50.0, 80.0, 65.0, 1.0) / 100
        
        with col2:
            immediate_exit_costs = st.slider("Immediate Exit Costs (%)", 0.0, 5.0, 2.6, 0.1) / 100
            future_exit_costs = st.slider("Future Exit Costs (%)", 0.0, 5.0, 1.8, 0.1) / 100
            financing_rate = st.slider("Annual Financing Rate (%)", 5.0, 15.0, 10.0, 0.5) / 100
        
        base_year = len(base_cash_flows)  # Index for 2025
    
    # Tab 2: Distribution Settings
    with tab2:
        st.header("Probability Distributions")
        
        # Function to create distribution input widgets
        def dist_input(name, default_type, min_val, mode_val, max_val, mean_val=None, std_val=None):
            st.subheader(name)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                dist_type = st.selectbox(
                    "Distribution Type",
                    ["PERT", "Triangular", "Uniform", "Normal"],
                    index=["PERT", "Triangular", "Uniform", "Normal"].index(default_type),
                    key=f"{name}_dist"
                )
            
            with col2:
                if dist_type in ["PERT", "Triangular"]:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        min_v = st.number_input("Min", value=min_val, key=f"{name}_min")
                    with c2:
                        mode_v = st.number_input("Most Likely", value=mode_val, key=f"{name}_mode")
                    with c3:
                        max_v = st.number_input("Max", value=max_val, key=f"{name}_max")
                    params = {"min": min_v, "mode": mode_v, "max": max_v}
                
                elif dist_type == "Uniform":
                    c1, c2 = st.columns(2)
                    with c1:
                        min_v = st.number_input("Min", value=min_val, key=f"{name}_min")
                    with c2:
                        max_v = st.number_input("Max", value=max_val, key=f"{name}_max")
                    params = {"min": min_v, "max": max_v}
                
                elif dist_type == "Normal":
                    c1, c2 = st.columns(2)
                    with c1:
                        mean_v = st.number_input("Mean", value=mean_val or mode_val, key=f"{name}_mean")
                    with c2:
                        std_v = st.number_input("Std Dev", value=std_val or (max_val-min_val)/6, key=f"{name}_std")
                    params = {"mean": mean_v, "std": std_v}
            
            return dist_type, params
        
        # Create distribution inputs
        immediate_exit_dist_type, immediate_exit_params = dist_input(
            "Immediate Exit Price ($)", "PERT", 29000000, 33000000, 37000000
        )
        
        future_exit_dist_type, future_exit_params = dist_input(
            "Future Exit Price ($)", "PERT", 90000000, 95000000, 100000000
        )
        
        dev_cost_dist_type, dev_cost_params = dist_input(
            "Development Cost ($)", "PERT", 32000000, 36000000, 40000000
        )
        
        # Special case for delay with cap
        st.subheader("Additional Delay (Years)")
        delay_dist_type = st.selectbox(
            "Distribution Type",
            ["LogNormal", "PERT", "Triangular", "Uniform", "Normal"],
            index=0,
            key="delay_dist"
        )
        
        if delay_dist_type == "LogNormal":
            c1, c2, c3 = st.columns(3)
            with c1:
                delay_mean = st.number_input("Mean", 0.0, 5.0, 0.5, 0.1, key="delay_mean")
            with c2:
                delay_95 = st.number_input("95th Percentile", 0.0, 10.0, 2.0, 0.1, key="delay_95")
            with c3:
                delay_cap = st.number_input("Cap Value", 0.0, 10.0, 2.0, 0.1, key="delay_cap")
            delay_params = {"mean": delay_mean, "percentile_95": delay_95, "cap": delay_cap}
        else:
            # Handle other distributions similar to above
            if delay_dist_type in ["PERT", "Triangular"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    delay_min = st.number_input("Min", 0.0, 5.0, 0.0, 0.1, key="delay_min")
                with c2:
                    delay_mode = st.number_input("Most Likely", 0.0, 5.0, 0.5, 0.1, key="delay_mode")
                with c3:
                    delay_max = st.number_input("Max", 0.0, 5.0, 2.0, 0.1, key="delay_max")
                delay_params = {"min": delay_min, "mode": delay_mode, "max": delay_max, "cap": 2.0}
            elif delay_dist_type == "Uniform":
                c1, c2 = st.columns(2)
                with c1:
                    delay_min = st.number_input("Min", 0.0, 5.0, 0.0, 0.1, key="delay_min")
                with c2:
                    delay_max = st.number_input("Max", 0.0, 5.0, 2.0, 0.1, key="delay_max")
                delay_params = {"min": delay_min, "max": delay_max, "cap": 2.0}
            elif delay_dist_type == "Normal":
                c1, c2 = st.columns(2)
                with c1:
                    delay_mean = st.number_input("Mean", 0.0, 5.0, 0.5, 0.1, key="delay_mean")
                with c2:
                    delay_std = st.number_input("Std Dev", 0.1, 2.0, 0.5, 0.1, key="delay_std")
                delay_params = {"mean": delay_mean, "std": delay_std, "cap": 2.0}
        
        # Collect distribution settings
        distribution_types = {
            'immediate_exit_price': immediate_exit_dist_type,
            'future_exit_price': future_exit_dist_type,
            'development_cost': dev_cost_dist_type,
            'additional_delay': delay_dist_type
        }
        
        distribution_params = {
            'immediate_exit_price': immediate_exit_params,
            'future_exit_price': future_exit_params,
            'development_cost': dev_cost_params,
            'additional_delay': delay_params
        }
        
        # Run simulation button
        if st.button("Run Monte Carlo Simulation", type="primary"):
            try:
                with st.spinner(f"Running {num_sims} simulations..."):
                    simulator = InvestmentSimulator(
                        cash_flows=base_cash_flows,
                        current_loan=current_loan,
                        dist_types=distribution_types,
                        dist_params=distribution_params,
                        base_year=base_year,
                        dev_time=standard_dev_time,
                        debt_pct=debt_pct,
                        financing_rate=financing_rate,
                        immediate_costs_pct=immediate_exit_costs,
                        future_costs_pct=future_exit_costs
                    )
                    
                    immediate_exit_df, continue_dev_df = simulator.run_simulation(num_sims)
                    st.session_state.results = (immediate_exit_df, continue_dev_df)
                    st.success(f"Simulation completed successfully!")
            except Exception as e:
                st.error(f"Error during simulation: {str(e)}")
    
    # Tab 3: Results
    with tab3:
        if st.session_state.results is None:
            st.info("Run the simulation to see results.")
        else:
            immediate_exit_df, continue_dev_df = st.session_state.results
            
            # Calculate statistics
            ie_mean = immediate_exit_df['IRR'].mean()
            ie_median = immediate_exit_df['IRR'].median()
            ie_5th = np.percentile(immediate_exit_df['IRR'], 5)
            ie_95th = np.percentile(immediate_exit_df['IRR'], 95)
            
            cd_mean = continue_dev_df['IRR'].mean()
            cd_median = continue_dev_df['IRR'].median()
            cd_5th = np.percentile(continue_dev_df['IRR'], 5)
            cd_95th = np.percentile(continue_dev_df['IRR'], 95)

            # Add Profit on Cost stats
            ie_profit_mean = immediate_exit_df['Profit_on_Cost'].mean()
            ie_profit_median = immediate_exit_df['Profit_on_Cost'].median()
            ie_profit_5th = np.percentile(immediate_exit_df['Profit_on_Cost'], 5)
            ie_profit_95th = np.percentile(immediate_exit_df['Profit_on_Cost'], 95)
        
            cd_profit_mean = continue_dev_df['Profit_on_Cost'].mean()
            cd_profit_median = continue_dev_df['Profit_on_Cost'].median()
            cd_profit_5th = np.percentile(continue_dev_df['Profit_on_Cost'], 5)
            cd_profit_95th = np.percentile(continue_dev_df['Profit_on_Cost'], 95)
            
            # Calculate probability
            prob_better = (continue_dev_df['IRR'] > ie_mean).mean() * 100
            
            # Display summary stats
            st.header("Summary Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Immediate Exit")
                st.write(f"Mean IRR: {ie_mean:.2f}%")
                st.write(f"Median IRR: {ie_median:.2f}%")
                st.write(f"5th-95th Range: {ie_5th:.2f}% - {ie_95th:.2f}%")
                st.write(f"Mean Profit on Cost: {ie_profit_mean:.2f}%")
                st.write(f"Median Profit on Cost: {ie_profit_median:.2f}%")
                st.write(f"5th-95th Profit Range: {ie_profit_5th:.2f}% - {ie_profit_95th:.2f}%")
            
            with col2:
                st.subheader("Continue Development")
                st.write(f"Mean IRR: {cd_mean:.2f}%")
                st.write(f"Median IRR: {cd_median:.2f}%")
                st.write(f"5th-95th Range: {cd_5th:.2f}% - {cd_95th:.2f}%")
                st.write(f"Mean Profit on Cost: {cd_profit_mean:.2f}%")
                st.write(f"Median Profit on Cost: {cd_profit_median:.2f}%")
                st.write(f"5th-95th Profit Range: {cd_profit_5th:.2f}% - {cd_profit_95th:.2f}%")
            
            st.info(f"Probability that Continue Development IRR > Immediate Exit Mean: {prob_better:.1f}%")
            
            # Create visualizations
            st.header("Visualizations")
            
            # Distribution comparison
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=immediate_exit_df['IRR'],
                name="Immediate Exit",
                opacity=0.7,
                nbinsx=40,
                histnorm='probability density'
            ))
            
            fig.add_trace(go.Histogram(
                x=continue_dev_df['IRR'],
                name="Continue Development",
                opacity=0.7,
                nbinsx=40,
                histnorm='probability density'
            ))
            
            fig.update_layout(
                title="IRR Distribution Comparison",
                xaxis_title="IRR (%)",
                yaxis_title="Density",
                barmode='overlay',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # New Profit on Cost histogram
            profit_fig = go.Figure()
            profit_fig.add_trace(go.Histogram(
                x=immediate_exit_df['Profit_on_Cost'],
                name="Immediate Exit",
                opacity=0.7,
                nbinsx=40,
                histnorm='probability density'
            ))

            profit_fig.add_trace(go.Histogram(
                x=continue_dev_df['Profit_on_Cost'],
                name="Continue Development",
                opacity=0.7,
                nbinsx=40,
                histnorm='probability density'
            ))
            profit_fig.update_layout(
                title="Profit on Cost Distribution Comparison",
                xaxis_title="Profit on Cost (%)",
                yaxis_title="Density",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(profit_fig, use_container_width=True)
            
            # Advanced analysis options
            st.header("Analysis Options")
            
            # Add a segmentation option
            analysis_type = st.radio(
                "Analysis Type",
                ["Basic Distribution", "Segmentation Analysis", "Risk Analysis", "Variable Distributions"]
            )
            
            if analysis_type == "Segmentation Analysis":
                # Allow user to select a variable to segment by
                segment_var = st.selectbox(
                    "Segment Continue Development by",
                    ["Exit_Year", "Construction_Cost", "Land_Price", "Financing_Cost",
                     "Debt", "Additional_Equity", "Total_Development_Cost", "Additional_Delay"]
                )
                
                # Determine the number of segments based on the variable
                if segment_var == "Exit_Year":
                    # Use actual discrete years
                    segments = sorted(continue_dev_df[segment_var].unique())
                    
                    # Create a color palette with discrete colors
                    color_scale = px.colors.qualitative.Plotly
                    if len(segments) > len(color_scale):
                        color_scale = px.colors.qualitative.Alphabet
                    
                    # Map each year to a color
                    color_map = {year: color_scale[i % len(color_scale)]
                                for i, year in enumerate(segments)}
                    
                    # Create a segment column
                    continue_dev_df['Segment'] = continue_dev_df[segment_var]
                    
                    # Convert exit year to calendar year for display
                    year_labels = {year: f"Year {2025 + (int(year) - base_year)}" for year in segments}
                    
                else:
                    # For continuous variables, create segments
                    num_segments = st.slider("Number of Segments", 2, 10, 5)
                    
                    # Calculate percentile cutoffs
                    cutoffs = [np.percentile(continue_dev_df[segment_var],
                                           100 * i / num_segments)
                              for i in range(num_segments + 1)]
                    
                    # Assign each row to a segment
                    segment_labels = []
                    for i, row in continue_dev_df.iterrows():
                        value = row[segment_var]
                        for j in range(num_segments):
                            if cutoffs[j] <= value < cutoffs[j+1]:
                                segment_labels.append(j)
                                break
                            elif j == num_segments-1:  # Last bin includes upper bound
                                segment_labels.append(j)
                    
                    continue_dev_df['Segment'] = segment_labels
                    
                    # Create labels for the legend
                    if segment_var in ['Construction_Cost', 'Land_Price', 'Financing_Cost',
                                      'Debt', 'Additional_Equity', 'Total_Development_Cost']:
                        # Format currency values
                        year_labels = {j: f"${cutoffs[j]/1000000:.1f}M-${cutoffs[j+1]/1000000:.1f}M"
                                      for j in range(num_segments)}
                    else:
                        # Format other values
                        year_labels = {j: f"{cutoffs[j]:.1f}-{cutoffs[j+1]:.1f}"
                                      for j in range(num_segments)}
                
                # Create the segmented histogram
                seg_fig = go.Figure()
                
                # Add immediate exit histogram
                seg_fig.add_trace(go.Histogram(
                    x=immediate_exit_df['IRR'],
                    name="Immediate Exit",
                    opacity=0.5,
                    nbinsx=40,
                    histnorm='probability density',
                    marker_color='gray'
                ))
                
                # Add a histogram for each segment
                if segment_var == "Exit_Year":
                    for year in segments:
                        subset = continue_dev_df[continue_dev_df['Segment'] == year]
                        seg_fig.add_trace(go.Histogram(
                            x=subset['IRR'],
                            name=year_labels[year],
                            opacity=0.7,
                            nbinsx=20,
                            histnorm='probability density',
                            marker_color=color_map[year]
                        ))
                else:
                    for j in range(num_segments):
                        subset = continue_dev_df[continue_dev_df['Segment'] == j]
                        seg_fig.add_trace(go.Histogram(
                            x=subset['IRR'],
                            name=year_labels[j],
                            opacity=0.7,
                            nbinsx=20,
                            histnorm='probability density'
                        ))
                
                seg_fig.update_layout(
                    title=f"IRR Distribution by {segment_var.replace('_', ' ')}",
                    xaxis_title="IRR (%)",
                    yaxis_title="Density",
                    barmode='overlay',
                    height=500
                )
                
                st.plotly_chart(seg_fig, use_container_width=True, key="segmented_histogram")
                
                # Add a scatter plot to show the relationship between the segment variable and IRR
                scatter_fig = px.scatter(
                    continue_dev_df,
                    x=segment_var,
                    y='IRR',
                    color='Segment',
                    opacity=0.7,
                    color_discrete_map=color_map if segment_var == "Exit_Year" else None,
                    labels={segment_var: segment_var.replace('_', ' '), 'IRR': 'IRR (%)'},
                    title=f"Relationship between {segment_var.replace('_', ' ')} and IRR"
                )
                
                # Add a trendline
                scatter_fig.update_layout(height=400)
                
                st.plotly_chart(scatter_fig, use_container_width=True, key="segment_scatter")
                
                # Add a summary table showing statistics by segment
                st.subheader("Summary Statistics by Segment")
                
                summary_data = []
                for segment in sorted(continue_dev_df['Segment'].unique()):
                    subset = continue_dev_df[continue_dev_df['Segment'] == segment]
                    summary_data.append({
                        'Segment': year_labels[segment] if segment in year_labels else segment,
                        'Count': len(subset),
                        'Mean IRR (%)': f"{subset['IRR'].mean():.2f}",
                        'Median IRR (%)': f"{subset['IRR'].median():.2f}",
                        '5th Percentile (%)': f"{np.percentile(subset['IRR'], 5):.2f}",
                        '95th Percentile (%)': f"{np.percentile(subset['IRR'], 95):.2f}",
                        'Probability > IE Mean (%)': f"{(subset['IRR'] > ie_mean).mean() * 100:.1f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)
                
            elif analysis_type == "Correlation Analysis":
                # Add correlation visualization
                corr = continue_dev_df.corr()['IRR'].drop('IRR').sort_values(ascending=False)
                corr_fig = px.bar(
                    x=corr.values, y=corr.index,
                    orientation='h',
                    labels={"x": "Correlation with IRR", "y": "Variable"},
                    title="Sensitivity Analysis"
                )
                st.plotly_chart(corr_fig, use_container_width=True, key="correlation_chart")
                
            elif analysis_type == "Risk Analysis":
                # Risk metrics
                min_irr = st.slider("Minimum Acceptable IRR (%)", 0.0, 30.0, 10.0, 0.5)
                
                ie_prob = (immediate_exit_df['IRR'] >= min_irr).mean() * 100
                cd_prob = (continue_dev_df['IRR'] >= min_irr).mean() * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probability IE ≥ Target", f"{ie_prob:.1f}%")
                with col2:
                    st.metric("Probability CD ≥ Target", f"{cd_prob:.1f}%")
                
                # Value at Risk
                var_level = st.slider("VaR Confidence (%)", 90, 99, 95, 1)
                
                ie_var = np.percentile(immediate_exit_df['IRR'], 100 - var_level)
                cd_var = np.percentile(continue_dev_df['IRR'], 100 - var_level)
                
                st.write(f"At {var_level}% confidence:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("IE IRR will not fall below", f"{ie_var:.2f}%")
                with col2:
                    st.metric("CD IRR will not fall below", f"{cd_var:.2f}%")
                
            elif analysis_type == "Variable Distributions":
                # Variable distribution plots
                var_to_plot = st.selectbox(
                    "Select Variable",
                    ["Future_Exit_Price", "Total_Development_Cost", "Additional_Delay", "Additional_Equity"]
                )
                
                var_fig = px.histogram(
                    continue_dev_df, x=var_to_plot,
                    marginal="box", nbins=40,
                    title=f"Distribution of {var_to_plot.replace('_', ' ')}"
                )
                
                st.plotly_chart(var_fig, use_container_width=True)
                
if __name__ == "__main__":
    run_dashboard()
