import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Time Series Analysis", layout="wide")

# Title
st.title("📊 Time Series Analysis & Business KPIs")

st.sidebar.header("1. Load Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Historical data file (CSV)", type=["csv"])

# --- DATA LOADING ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Convert date if 'dteday' column exists (common in these datasets)
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'])

        st.session_state['df_condiviso'] = df
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")

# --- ANALYSIS ---
if 'df_condiviso' in st.session_state:
    df = st.session_state['df_condiviso']

    # ---------------------------------------------------------
    # SECTION 1: OVERVIEW
    # ---------------------------------------------------------
    st.write(f"Dataset loaded: {len(df)} rows.")

    with st.expander("View Data Preview and Basic Statistics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
        with col2:
            st.write("**Available columns:**", df.columns.tolist())

        st.dataframe(df.head())

        # Quick statistics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Average (cnt)", f"{df['cnt'].mean():.0f}")
        c2.metric("Min", f"{df['cnt'].min()}")
        c3.metric("Max", f"{df['cnt'].max()}")
        c4.metric("Std Dev", f"{df['cnt'].std():.0f}")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 2: KPI & DECISION SUPPORT (Index Date + OHE)
    # ---------------------------------------------------------

    st.header("🚀 KPI & Decision Support")

    # Column names
    col_registered = 'registered'
    col_total = 'cnt'
    # Weather One-Hot columns (in order of quality)
    cols_meteo_ohe = ['weathersit_1.0', 'weathersit_2.0', 'weathersit_3.0', 'weathersit_4.0']

    if col_registered in df.columns and col_total in df.columns:
        
       # --- 0. INDEX PREPARATION ---
        # Ensure the index is datetime.
        try:
            # 1. Set the first column as index (datetime)
            df.index = pd.to_datetime(df.iloc[:, 0])

            # 2. Rename the index to avoid generic names like "Unnamed: 0"
            df.index.name = 'Date_Index'

            # 3. REMOVE the first column from data to avoid duplicates/ambiguity
            #    (Select all rows and all columns from the second onwards)
            df = df.iloc[:, 1:]

        except Exception as e:
            st.error(f"Error handling index: {e}")

        # --- 1. DAILY AGGREGATION (Using the newly set index) ---
        # Group by summing volumes using the index date
        # df.index.date takes only the "YYYY-MM-DD" part ignoring the time
        df_daily = df.groupby(df.index.date)[[col_registered, col_total]].sum()

        # Convert the new df_daily index back to datetime for safety (for Plotly)
        df_daily.index = pd.to_datetime(df_daily.index)

        # Calculate Efficiency KPI on DAILY data
        df_daily['kpi_efficiency_rate'] = (df_daily[col_registered] / df_daily[col_total]) * 100

        st.success(f"✅ Data aggregated by day using the first column as date ({len(df_daily)} days found).")

        # --- 2. KPI CARDS ---
        kpi_avg = df_daily['kpi_efficiency_rate'].mean()
        kpi_last = df_daily['kpi_efficiency_rate'].iloc[-1]

        k1, k2 = st.columns(2)
        k1.metric("Daily Average Efficiency", f"{kpi_avg:.1f}%")
        k2.metric("Last Day Efficiency", f"{kpi_last:.1f}%", delta=f"{kpi_last - kpi_avg:.1f}%")

        # --- 3. TIME MONITORING CHART ---
        st.subheader("📈 Daily Trend: % Registered")
        
        fig_kpi_time = go.Figure()
        
        fig_kpi_time.add_trace(go.Scatter(
            x=df_daily.index,  # Use the index (dates) as X axis
            y=df_daily['kpi_efficiency_rate'],
            mode='lines',
            name='% Registered',
            line=dict(color='#00CC96', width=2)
        ))

        fig_kpi_time.add_hline(y=kpi_avg, line_dash="dot", annotation_text="Average", line_color="red")

        fig_kpi_time.update_layout(
            title="Business Efficiency Trend (Aggregated by Day)",
            template="plotly_white",
            yaxis_title="% Registered",
            hovermode="x unified"
        )
        st.plotly_chart(fig_kpi_time, width='stretch')

        # ---------------------------------------------------------
        # NEW SECTION: SUBSCRIBERS VS CASUAL ANALYSIS
        # ---------------------------------------------------------
        st.subheader("👥 Subscribers vs Casual Users")

        # Need casual column for this analysis
        if 'casual' in df.columns:
            # Add casual to daily aggregation
            df_daily_users = df.groupby(df.index.date)[['registered', 'casual', 'cnt']].sum()
            df_daily_users.index = pd.to_datetime(df_daily_users.index)

            # Calculate percentages
            df_daily_users['pct_registered'] = (df_daily_users['registered'] / df_daily_users['cnt']) * 100
            df_daily_users['pct_casual'] = (df_daily_users['casual'] / df_daily_users['cnt']) * 100

            # Overall metrics
            avg_reg = df_daily_users['pct_registered'].mean()
            avg_cas = df_daily_users['pct_casual'].mean()

            col_u1, col_u2, col_u3, col_u4 = st.columns(4)
            col_u1.metric("Avg % Registered", f"{avg_reg:.1f}%")
            col_u2.metric("Avg % Casual", f"{avg_cas:.1f}%")
            col_u3.metric("Total Registered", f"{df_daily_users['registered'].sum():,.0f}")
            col_u4.metric("Total Casual", f"{df_daily_users['casual'].sum():,.0f}")

            # Stacked area chart showing user composition over time
            fig_users = go.Figure()
            fig_users.add_trace(go.Scatter(
                x=df_daily_users.index,
                y=df_daily_users['registered'],
                fill='tozeroy',
                name='Registered',
                line=dict(color='#636EFA'),
                fillcolor='rgba(99, 110, 250, 0.6)'
            ))
            fig_users.add_trace(go.Scatter(
                x=df_daily_users.index,
                y=df_daily_users['registered'] + df_daily_users['casual'],
                fill='tonexty',
                name='Casual',
                line=dict(color='#EF553B'),
                fillcolor='rgba(239, 85, 59, 0.6)'
            ))
            fig_users.update_layout(
                title="Daily User Composition: Registered vs Casual",
                xaxis_title="Date",
                yaxis_title="Number of Rentals",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_users, width='stretch')

            # Monthly breakdown
            df_monthly = df.copy()
            df_monthly['month'] = df_monthly.index.to_period('M').astype(str)
            monthly_users = df_monthly.groupby('month')[['registered', 'casual']].sum().reset_index()

            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(x=monthly_users['month'], y=monthly_users['registered'], name='Registered', marker_color='#636EFA'))
            fig_monthly.add_trace(go.Bar(x=monthly_users['month'], y=monthly_users['casual'], name='Casual', marker_color='#EF553B'))
            fig_monthly.update_layout(
                title="Monthly User Distribution",
                barmode='group',
                xaxis_title="Month",
                yaxis_title="Total Rentals",
                template="plotly_white"
            )
            st.plotly_chart(fig_monthly, width='stretch')

            st.info(f"""
            💡 **Key Insight:** On average, **{avg_reg:.1f}%** of users are registered subscribers
            while **{avg_cas:.1f}%** are casual users. Registered users provide more stable, predictable demand.
            """)
        else:
            st.warning("Column 'casual' not found for user composition analysis.")

        st.markdown("---")

                # --- 4. WEATHER HANDLING (COMBO CHART + INTERACTIVE INSIGHT) ---
        cols_esistenti = [c for c in cols_meteo_ohe if c in df.columns]

        if len(cols_esistenti) > 0:
            st.subheader("🌤️ Weather Impact: 2011 vs 2012 Trend Analysis")

            # --- METRIC SELECTOR ---
            scelta_utente = st.radio(
                "Choose the user type to analyze:",
                ["Total (cnt)", "Registered (registered)", "Casual (casual)"],
                horizontal=True,
                key="radio_weather_combo"
            )

            map_colonne = {
                "Total (cnt)": "cnt",
                "Registered (registered)": "registered",
                "Casual (casual)": "casual"
            }
            col_analizzata = map_colonne[scelta_utente]

            if col_analizzata not in df.columns:
                st.error(f"Column '{col_analizzata}' is not present in the dataset.")
            else:
                # Data Preparation
                df_meteo = df.copy()
                df_meteo['Year'] = df_meteo.index.year
                df_meteo['Meteo_Raw'] = df_meteo[cols_esistenti].idxmax(axis=1)

                mapping_nomi = {
                    'weathersit_1.0': '1. Sunny/Clear',
                    'weathersit_2.0': '2. Cloudy',
                    'weathersit_3.0': '3. Light Rain',
                    'weathersit_4.0': '4. Storm'
                }
                df_meteo['Meteo_Label'] = df_meteo['Meteo_Raw'].map(mapping_nomi)

                # Grouping
                df_weather_comp = df_meteo.groupby(['Year', 'Meteo_Label'])[col_analizzata].mean().reset_index()
                
                df_2011 = df_weather_comp[df_weather_comp['Year'] == 2011]
                df_2012 = df_weather_comp[df_weather_comp['Year'] == 2012]

                # --- COMBO CHART CONSTRUCTION ---
                fig_weather = go.Figure()

                # 2011: Bars + Line
                fig_weather.add_trace(go.Bar(
                    x=df_2011['Meteo_Label'],
                    y=df_2011[col_analizzata],
                    name='2011 (Volume)',
                    marker_color='rgba(99, 110, 250, 0.5)', # Semi-transparent blue
                    text=df_2011[col_analizzata].apply(lambda x: f"{x:.0f}"),
                    textposition='auto'
                ))
                fig_weather.add_trace(go.Scatter(
                    x=df_2011['Meteo_Label'],
                    y=df_2011[col_analizzata],
                    name='2011 (Trend)',
                    mode='lines+markers',
                    marker=dict(symbol='circle', size=8, color='#636EFA'),
                    line=dict(width=3, color='#636EFA')
                ))

                # 2012: Bars + Line
                fig_weather.add_trace(go.Bar(
                    x=df_2012['Meteo_Label'],
                    y=df_2012[col_analizzata],
                    name='2012 (Volume)',
                    marker_color='rgba(239, 85, 59, 0.5)', # Semi-transparent red
                    text=df_2012[col_analizzata].apply(lambda x: f"{x:.0f}"),
                    textposition='auto'
                ))
                fig_weather.add_trace(go.Scatter(
                    x=df_2012['Meteo_Label'],
                    y=df_2012[col_analizzata],
                    name='2012 (Trend)',
                    mode='lines+markers',
                    marker=dict(symbol='circle', size=8, color='#EF553B'),
                    line=dict(width=3, color='#EF553B')
                ))

                fig_weather.update_layout(
                    title=f"Weather Trend Analysis: {scelta_utente}",
                    xaxis_title="Weather Condition",
                    yaxis_title=f"Average {scelta_utente}",
                    barmode='group',
                    template="plotly_white",
                    legend=dict(title="Year/Type"),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_weather, width='stretch')
                
                # --- INTERACTIVE INSIGHT ---
                st.markdown("#### 📉 Weather Sensitivity Calculation")

                # Specific selector for insight
                anno_insight = st.radio(
                    "For which year do you want to calculate the demand drop (Sunny vs Storm)?",
                    [2011, 2012],
                    horizontal=True,
                    key="radio_insight_year"
                )

                # Select the dataframe for the chosen year
                df_target = df_2011 if anno_insight == 2011 else df_2012

                if not df_target.empty:
                    # Search for specific values using partial string matching ('1.' for Sunny, '4.' for Storm)
                    val_sole = df_target[df_target['Meteo_Label'].str.contains('1.')][col_analizzata].values
                    val_pioggia = df_target[df_target['Meteo_Label'].str.contains('4.')][col_analizzata].values

                    if len(val_sole) > 0 and len(val_pioggia) > 0:
                        val_s = val_sole[0]
                        val_p = val_pioggia[0]

                        if val_s > 0:
                            calo = ((val_s - val_p) / val_s) * 100

                            # Create a dynamic message
                            st.info(f"""
                            💡 **Insight {anno_insight}:** In **{anno_insight}**, going from optimal conditions (Sunny) to storm,
                            the demand for **{scelta_utente}** drops by **{calo:.1f}%**.

                            * Average with Sunny: **{val_s:.0f}**
                            * Average with Storm: **{val_p:.0f}**
                            """)
                        else:
                            st.warning("The average value with Sunny is 0, cannot calculate percentage.")
                    else:
                        st.warning(f"Insufficient data in {anno_insight} to compare Sunny vs Storm.")
                else:
                    st.warning("No data available for the selected year.")

                # ---------------------------------------------------------
                # NEW: DUAL-USER WEATHER IMPACT
                # ---------------------------------------------------------
                st.markdown("#### 🌧️ Climate Impact by User Type")

                if 'casual' in df.columns and 'registered' in df.columns:
                    # Calculate weather impact for both user types
                    df_weather_users = df_meteo.groupby('Meteo_Label')[['registered', 'casual']].mean().reset_index()

                    # Get sunny and storm values for each user type
                    sunny_reg = df_weather_users[df_weather_users['Meteo_Label'].str.contains('1.')]['registered'].values
                    storm_reg = df_weather_users[df_weather_users['Meteo_Label'].str.contains('4.')]['registered'].values
                    sunny_cas = df_weather_users[df_weather_users['Meteo_Label'].str.contains('1.')]['casual'].values
                    storm_cas = df_weather_users[df_weather_users['Meteo_Label'].str.contains('4.')]['casual'].values

                    if len(sunny_reg) > 0 and len(storm_reg) > 0 and len(sunny_cas) > 0 and len(storm_cas) > 0:
                        drop_registered = ((sunny_reg[0] - storm_reg[0]) / sunny_reg[0]) * 100
                        drop_casual = ((sunny_cas[0] - storm_cas[0]) / sunny_cas[0]) * 100

                        col_w1, col_w2 = st.columns(2)
                        col_w1.metric(
                            "Registered Users Drop (Storm)",
                            f"{drop_registered:.1f}%",
                            delta=f"-{storm_reg[0]:.0f} from {sunny_reg[0]:.0f}",
                            delta_color="inverse"
                        )
                        col_w2.metric(
                            "Casual Users Drop (Storm)",
                            f"{drop_casual:.1f}%",
                            delta=f"-{storm_cas[0]:.0f} from {sunny_cas[0]:.0f}",
                            delta_color="inverse"
                        )

                        # Comparative bar chart
                        fig_impact = go.Figure()
                        weather_conditions = ['Sunny', 'Storm']
                        fig_impact.add_trace(go.Bar(
                            x=weather_conditions,
                            y=[sunny_reg[0], storm_reg[0]],
                            name='Registered',
                            marker_color='#636EFA',
                            text=[f'{sunny_reg[0]:.0f}', f'{storm_reg[0]:.0f}'],
                            textposition='auto'
                        ))
                        fig_impact.add_trace(go.Bar(
                            x=weather_conditions,
                            y=[sunny_cas[0], storm_cas[0]],
                            name='Casual',
                            marker_color='#EF553B',
                            text=[f'{sunny_cas[0]:.0f}', f'{storm_cas[0]:.0f}'],
                            textposition='auto'
                        ))
                        fig_impact.update_layout(
                            title="Weather Impact: Registered vs Casual Users",
                            xaxis_title="Weather Condition",
                            yaxis_title="Average Rentals",
                            barmode='group',
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_impact, width='stretch')

                        st.success(f"""
                        💡 **Key Finding:** Casual users are **much more weather-sensitive** than registered users!

                        - **Casual users** drop by **{drop_casual:.1f}%** in storm conditions
                        - **Registered users** drop by only **{drop_registered:.1f}%** in storm conditions

                        This suggests registered users (commuters) have **essential trips** regardless of weather,
                        while casual users (recreational) **avoid bad weather**.
                        """)
                    else:
                        st.warning("Insufficient weather data for dual-user comparison.")
                else:
                    st.warning("Columns 'registered' or 'casual' not found for dual-user analysis.")

        else:
            st.info("No weather columns found.")
    else:
        st.warning(f"Warning: Columns '{col_registered}' or '{col_total}' not found.")
    # ---------------------------------------------------------
    # SECTION 3: CORRELATION ANALYSIS
    # ---------------------------------------------------------
    st.subheader("🔎 In-Depth Analysis (Filters & Correlations)")

    col_workday = 'workingday'
    if col_workday in df.columns:
        filtro_tipo = st.radio(
            "Filter data by working days:",
            options=["All data", "Working Days (1)", "Non-Working Days (0)"],
            horizontal=True,
            key="filtro_workday" # Added key to avoid conflicts
        )
        if filtro_tipo == "Working Days (1)":
            df_filtrato = df[df[col_workday] == 1]
        elif filtro_tipo == "Non-Working Days (0)":
            df_filtrato = df[df[col_workday] == 0]
        else:
            df_filtrato = df
    else:
        df_filtrato = df

    opzioni_analisi = ['hr', 'season', 'temp', 'hum', 'windspeed']
    opzioni_disponibili = [col for col in opzioni_analisi if col in df.columns]

    if opzioni_disponibili:
        variabile_x = st.selectbox("Choose X variable:", opzioni_disponibili, key="var_select")

        tab1, tab2 = st.tabs(["🔴 Scatter with Average", "📦 Box Plot"])

        with tab1:
            df_media = df_filtrato.groupby(variabile_x)['cnt'].mean().reset_index().sort_values(by=variabile_x)
            fig_scatter = px.scatter(df_filtrato, x=variabile_x, y="cnt", color="cnt", opacity=0.4, title=f"Scatter: {variabile_x} vs cnt")
            fig_scatter.add_scatter(x=df_media[variabile_x], y=df_media['cnt'], mode='markers', name='Average', marker=dict(color='red', size=10, symbol='diamond'))
            st.plotly_chart(fig_scatter, width='stretch')

        with tab2:
            fig_box = px.box(df_filtrato, x=variabile_x, y="cnt", color=variabile_x, title=f"Box Plot: {variabile_x}")
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, width='stretch')

    # ---------------------------------------------------------
    # NEW SECTION: PEAK HOURS ANALYSIS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🕐 Peak Hours Analysis")

    if 'hr' in df.columns:
        # Create heatmap: Hour x Day of Week
        df_heatmap = df.copy()
        df_heatmap['dayofweek'] = df_heatmap.index.dayofweek

        heatmap_data = df_heatmap.pivot_table(
            values='cnt',
            index='hr',
            columns='dayofweek',
            aggfunc='mean'
        )

        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Day of Week", y="Hour", color="Avg Rentals"),
            x=day_labels,
            y=list(range(24)),
            aspect="auto",
            color_continuous_scale='YlOrRd'
        )
        fig_heatmap.update_layout(
            title="Average Bike Rentals: Hour vs Day of Week",
            template="plotly_white"
        )
        st.plotly_chart(fig_heatmap, width='stretch')

        # Peak hours identification
        hourly_avg = df.groupby('hr')['cnt'].mean()
        peak_hour = hourly_avg.idxmax()
        peak_value = hourly_avg.max()
        off_peak_hour = hourly_avg.idxmin()
        off_peak_value = hourly_avg.min()

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Peak Hour", f"{peak_hour}:00", f"{peak_value:.0f} avg rentals")
        col_p2.metric("Off-Peak Hour", f"{off_peak_hour}:00", f"{off_peak_value:.0f} avg rentals")
        col_p3.metric("Morning Rush (7-9)", f"{hourly_avg.loc[7:9].mean():.0f}")
        col_p4.metric("Evening Rush (17-19)", f"{hourly_avg.loc[17:19].mean():.0f}")

        # ---------------------------------------------------------
        # WORKING DAY VS NON-WORKING DAY COMPARISON
        # ---------------------------------------------------------
        st.markdown("#### 📊 Working Days vs Non-Working Days Hourly Pattern")

        if 'workingday' in df.columns:
            df_working = df[df['workingday'] == 1].groupby('hr')['cnt'].mean()
            df_nonworking = df[df['workingday'] == 0].groupby('hr')['cnt'].mean()

            fig_comparison = go.Figure()

            # Working days line
            fig_comparison.add_trace(go.Scatter(
                x=df_working.index,
                y=df_working.values,
                name='Working Days',
                mode='lines+markers',
                line=dict(color='#636EFA', width=3),
                marker=dict(size=8)
            ))

            # Non-working days line
            fig_comparison.add_trace(go.Scatter(
                x=df_nonworking.index,
                y=df_nonworking.values,
                name='Non-Working Days',
                mode='lines+markers',
                line=dict(color='#EF553B', width=3),
                marker=dict(size=8)
            ))

            # Add shaded regions for commute hours
            fig_comparison.add_vrect(x0=6, x1=9, fillcolor="rgba(0,100,80,0.1)",
                                     layer="below", line_width=0,
                                     annotation_text="Morning Rush", annotation_position="top left")
            fig_comparison.add_vrect(x0=16, x1=19, fillcolor="rgba(0,100,80,0.1)",
                                     layer="below", line_width=0,
                                     annotation_text="Evening Rush", annotation_position="top left")

            fig_comparison.update_layout(
                title="Hourly Rental Patterns: Working vs Non-Working Days",
                xaxis_title="Hour of Day",
                yaxis_title="Average Rentals",
                template="plotly_white",
                hovermode="x unified",
                xaxis=dict(tickmode='linear', tick0=0, dtick=2)
            )
            st.plotly_chart(fig_comparison, width='stretch')

            st.info("""
            💡 **Key Insight:** Working days show clear **commute peaks** at 8 AM and 5-6 PM (workers using bikes for transportation).
            Non-working days show a **gradual increase** peaking around midday (recreational use).
            """)
        else:
            st.warning("Column 'workingday' not found for comparison.")
    else:
        st.warning("Column 'hr' not found for peak hours analysis.")

else:
    st.info("👈 Load a CSV file from the sidebar to get started.")
