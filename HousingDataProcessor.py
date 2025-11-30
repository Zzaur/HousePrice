
"""Import Libraries"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox1p
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize


class HousingDataProcessor:
    """
    Comprehensive housing data quality assessment and feature engineering pipeline
    """

    def __init__(self, df):
        self.df = df.copy()
        self.outlier_report = {}
        self.missing_report = {}

    # MISSING VALUE ANALYSIS 

    def analyze_missing_values(self):
        """Analyze missing values and categorize by pattern"""
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

        self.missing_report = {
            'Low (<5%)': missing_pct[missing_pct < 5],
            'Medium (5-25%)': missing_pct[(missing_pct >= 5) & (missing_pct <= 25)],
            'High (>25%)': missing_pct[missing_pct > 25]
        }

        print("=" * 60)
        print("MISSING VALUE ANALYSIS")
        print("=" * 60)
        for category, features in self.missing_report.items():
            if len(features) > 0:
                print(f"\n{category}:")
                for feat, pct in features.items():
                    print(f"  {feat}: {pct:.2f}%")

        return self.missing_report

    def handle_missing_values(self):
        """Handle missing values based on feature type"""
        df = self.df.copy()

        # Systematically missing (luxury features) - means "absent"
        luxury_features = ['PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
                          'Alley', 'FireplaceQu']
        for feat in luxury_features:
            if feat in df.columns:
                if df[feat].dtype in ['float64', 'int64']:
                    df[feat] = df[feat].fillna(0)
                    df[f'Has_{feat}'] = (df[feat] > 0).astype(int)
                else:
                    df[feat] = df[feat].fillna('None')
                    df[f'Has_{feat}'] = (df[feat] != 'None').astype(int)

        # Garage features - missing means no garage
        garage_features = ['GarageType', 'GarageFinish', 'GarageQual',
                          'GarageCond', 'GarageYrBlt', 'GarageCars', 'GarageArea']
        for feat in garage_features:
            if feat in df.columns:
                if df[feat].dtype in ['float64', 'int64']:
                    df[feat] = df[feat].fillna(0)
                else:
                    df[feat] = df[feat].fillna('None')
        if 'GarageCars' in df.columns:
            df['Has_Garage'] = (df['GarageCars'] > 0).astype(int)

        # Basement features - missing means no basement
        basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1',
                            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                            'BsmtFullBath', 'BsmtHalfBath']
        for feat in basement_features:
            if feat in df.columns:
                if df[feat].dtype in ['float64', 'int64']:
                    df[feat] = df[feat].fillna(0)
                else:
                    df[feat] = df[feat].fillna('None')
        if 'TotalBsmtSF' in df.columns:
            df['Has_Basement'] = (df['TotalBsmtSF'] > 0).astype(int)

        # Masonry veneer
        if 'MasVnrType' in df.columns:
            df['MasVnrType'] = df['MasVnrType'].fillna('None')
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
            df['Has_MasVnr'] = (df['MasVnrArea'] > 0).astype(int)

        # LotFrontage - impute by neighborhood median
        if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median()))

        # Remaining categorical - use mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        # Remaining numerical - use median
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        self.df = df
        print(f"\n✓ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df

    # OUTLIER DETECTION

    def detect_outliers(self, features=None, methods=['iqr', 'zscore']):
        """Detect outliers using multiple methods"""
        df = self.df.copy()

        if features is None:
            features = df.select_dtypes(include=['int64', 'float64']).columns

        outlier_indices = {}

        for feature in features:
            if feature not in df.columns:
                continue

            outliers = set()

            # IQR Method
            if 'iqr' in methods:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                iqr_outliers = df[(df[feature] < lower) | (df[feature] > upper)].index
                outliers.update(iqr_outliers)

            # Z-Score Method
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(df[feature].dropna()))
                z_outliers = df[feature].dropna()[z_scores > 3].index
                outliers.update(z_outliers)

            if len(outliers) > 0:
                outlier_indices[feature] = list(outliers)

        self.outlier_report = outlier_indices

        print("\n" + "=" * 60)
        print("OUTLIER DETECTION REPORT")
        print("=" * 60)
        for feature, indices in outlier_indices.items():
            pct = len(indices) / len(df) * 100
            print(f"{feature}: {len(indices)} outliers ({pct:.2f}%)")

        return outlier_indices

    def detect_domain_outliers(self):
        """Detect outliers based on domain-specific rules"""
        df = self.df.copy()
        domain_outliers = {}

        rules = {
            'SalePrice': (10000, 50000000),
            'GrLivArea': (200, 15000),
            'LotArea': (1000, 200000),
            'YearBuilt': (1800, 2025),
            'BedroomAbvGr': (0, 10),
            'TotRmsAbvGrd': (1, 15)
        }

        for feature, (lower, upper) in rules.items():
            if feature in df.columns:
                outliers = df[(df[feature] < lower) | (df[feature] > upper)].index
                if len(outliers) > 0:
                    domain_outliers[feature] = list(outliers)

        print("\n" + "=" * 60)
        print("DOMAIN-SPECIFIC OUTLIERS")
        print("=" * 60)
        for feature, indices in domain_outliers.items():
            print(f"{feature}: {len(indices)} violations")

        return domain_outliers

    def create_market_segments(self, price_col='SalePrice',
                               thresholds=[500000, 1000000]):
        """
        Create market segments for separate modeling

        Args:
            price_col: Column containing sale prices
            thresholds: Price thresholds for segmentation

        Returns:
            Dictionary of segmented dataframes
        """
        df = self.df.copy()

        if price_col not in df.columns:
            print(f"Error: {price_col} not found in data")
            return None

        segments = {}

        # Create segment labels
        df['MarketSegment'] = pd.cut(df[price_col],bins=[0] + thresholds + [float('inf')],labels=['Mainstream', 'Upper', 'Luxury'])

        # Split into separate datasets
        segments['mainstream'] = df[df['MarketSegment'] == 'Mainstream'].copy()
        segments['upper'] = df[df['MarketSegment'] == 'Upper'].copy()
        segments['luxury'] = df[df['MarketSegment'] == 'Luxury'].copy()
        segments['full'] = df

        print("\n" + "=" * 60)
        print("MARKET SEGMENTATION")
        print("=" * 60)
        print(f"Mainstream Market (<${thresholds[0]:,}): {len(segments['mainstream'])} homes")
        print(f"  Mean Price: ${segments['mainstream'][price_col].mean():,.2f}")
        print(f"  Median Price: ${segments['mainstream'][price_col].median():,.2f}")

        print(f"\nUpper Market (${thresholds[0]:,}-${thresholds[1]:,}): {len(segments['upper'])} homes")
        print(f"  Mean Price: ${segments['upper'][price_col].mean():,.2f}")
        print(f"  Median Price: ${segments['upper'][price_col].median():,.2f}")

        print(f"\nLuxury Market (>${thresholds[1]:,}): {len(segments['luxury'])} homes")
        print(f"  Mean Price: ${segments['luxury'][price_col].mean():,.2f}")
        print(f"  Median Price: ${segments['luxury'][price_col].median():,.2f}")

        print("\n" + "=" * 60)
        print("SEGMENT CHARACTERISTICS")
        print("=" * 60)

        # Analyze key differences between segments
        key_features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBath', 'GarageCars']
        available_features = [f for f in key_features if f in df.columns]

        if available_features:
            print("\nAverage values by segment:")
            for segment_name, segment_df in [('Mainstream', segments['mainstream']),
                                             ('Upper', segments['upper']),
                                             ('Luxury', segments['luxury'])]:
                print(f"\n{segment_name}:")
                for feature in available_features:
                    avg_val = segment_df[feature].mean()
                    print(f"  {feature}: {avg_val:.1f}")

        self.segments = segments
        return segments

    def cap_outliers(self, features, percentile=99, keep_original=True):
        """
        Cap outliers at specified percentile

        Args:
            features: List of features to cap
            percentile: Percentile to cap at (default 99)
            keep_original: If True, keeps original feature + creates capped version
        """
        df = self.df.copy()

        for feature in features:
            if feature in df.columns:
                upper = df[feature].quantile(percentile / 100)
                lower = df[feature].quantile((100 - percentile) / 100)

                if keep_original:
                    # Keep original + create capped version
                    df[f'{feature}_Capped'] = df[feature].clip(lower=lower, upper=upper)
                    print(f"  Created: {feature}_Capped (original preserved)")
                else:
                    # Replace original
                    df[feature] = df[feature].clip(lower=lower, upper=upper)
                    print(f"  Capped: {feature}")

        self.df = df
        print(f"\n✓ Outliers capped at {percentile}th percentile for {len(features)} features")
        return df

    # UNIVARIATE ANALYSIS

    def analyze_target(self, target='SalePrice'):
        """Analyze target variable distribution"""
        if target not in self.df.columns:
            print(f"Target '{target}' not found in data")
            return

        print("\n" + "=" * 60)
        print(f"TARGET VARIABLE ANALYSIS: {target}")
        print("=" * 60)

        data = self.df[target]

        print(f"Mean: ${data.mean():,.2f}")
        print(f"Median: ${data.median():,.2f}")
        print(f"Std Dev: ${data.std():,.2f}")
        print(f"Skewness: {data.skew():.3f}")
        print(f"Kurtosis: {data.kurtosis():.3f}")
        print(f"Min: ${data.min():,.2f}")
        print(f"Max: ${data.max():,.2f}")
        print(f"\nDistribution: {'Right-skewed' if data.skew() > 0.5 else 'Normal-ish'}")

        return {
            'mean': data.mean(),
            'median': data.median(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }

    def transform_target(self, target='SalePrice', method='log'):
        """Transform target variable"""
        df = self.df.copy()

        if method == 'log':
            df[f'{target}_Log'] = np.log1p(df[target])
            print(f"✓ Log transformation applied: {target}_Log")
        elif method == 'boxcox':
            df[f'{target}_BoxCox'], _ = stats.boxcox(df[target] + 1)
            print(f"✓ Box-Cox transformation applied: {target}_BoxCox")

        self.df = df
        return df

    # FEATURE ENGINEERING

    def create_size_features(self):
        """Create size-related features"""
        df = self.df.copy()

        # Lot coverage ratio
        if 'GrLivArea' in df.columns and 'LotArea' in df.columns:
            df['LotCoverage'] = df['GrLivArea'] / df['LotArea']

        # Above ground ratio
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            total_area = df['GrLivArea'] + df['TotalBsmtSF']
            df['AboveGroundRatio'] = df['GrLivArea'] / (total_area + 1)

        # Total square footage
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']

        # Total bathrooms
        if 'FullBath' in df.columns and 'HalfBath' in df.columns:
            df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
            if 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
                df['TotalBath'] += df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

        # Total porch area
        porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                      '3SsnPorch', 'ScreenPorch']
        available_porch = [col for col in porch_cols if col in df.columns]
        if available_porch:
            df['TotalPorchSF'] = df[available_porch].sum(axis=1)

        self.df = df
        print("✓ Size features created")
        return df

    def create_age_features(self):
        """Create age and condition features"""
        df = self.df.copy()
        current_year = 2025

        # Property age
        if 'YearBuilt' in df.columns:
            df['PropertyAge'] = current_year - df['YearBuilt']

        # Years since remodel
        if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
            df['YearsSinceRemodel'] = current_year - df['YearRemodAdd']
            df['WasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

        # Effective age (accounting for renovation)
        if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
            df['EffectiveAge'] = current_year - df[['YearBuilt', 'YearRemodAdd']].max(axis=1)

        # Age categories
        if 'PropertyAge' in df.columns:
            df['AgeCategory'] = pd.cut(df['PropertyAge'],
                                       bins=[-1, 5, 20, 50, 200],
                                       labels=['New', 'Modern', 'Mature', 'Historic'])

        # Garage age
        if 'GarageYrBlt' in df.columns:
            df['GarageAge'] = current_year - df['GarageYrBlt']
            df['GarageAge'] = df['GarageAge'].clip(lower=0)

        self.df = df
        print("✓ Age features created")
        return df

    def create_room_features(self):
        """Create room configuration features"""
        df = self.df.copy()

        # Bedroom to bathroom ratio
        if 'BedroomAbvGr' in df.columns and 'TotalBath' in df.columns:
            df['BedBathRatio'] = df['BedroomAbvGr'] / (df['TotalBath'] + 1)

        # Room density
        if 'TotRmsAbvGrd' in df.columns and 'GrLivArea' in df.columns:
            df['RoomDensity'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + 1)

        # Average room size
        if 'GrLivArea' in df.columns and 'TotRmsAbvGrd' in df.columns:
            df['AvgRoomSize'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)

        self.df = df
        print("✓ Room features created")
        return df

    def create_quality_features(self):
        """Create quality interaction features"""
        df = self.df.copy()

        # Overall quality score
        if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
            df['QualityScore'] = df['OverallQual'] * df['OverallCond']

        # Quality per square foot
        if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
            df['QualPerSF'] = df['OverallQual'] / (df['GrLivArea'] / 1000)

        self.df = df
        print("✓ Quality features created")
        return df

    def create_temporal_features(self):
        """Create temporal features"""
        df = self.df.copy()

        if 'MoSold' in df.columns:
            # Seasonal indicators (fixed bins)
            def assign_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:  # 9, 10, 11
                    return 'Fall'

            df['Season'] = df['MoSold'].apply(assign_season)

            # Cyclical encoding
            df['MonthSin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
            df['MonthCos'] = np.cos(2 * np.pi * df['MoSold'] / 12)

            # Peak season indicator
            df['IsPeakSeason'] = df['MoSold'].isin([5, 6, 7, 8]).astype(int)

        self.df = df
        print("✓ Temporal features created")
        return df

    def create_boolean_features(self):
        """Create binary indicator features"""
        df = self.df.copy()

        # Has second floor
        if '2ndFlrSF' in df.columns:
            df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

        # Has fireplace
        if 'Fireplaces' in df.columns:
            df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

        # Has porch
        if 'TotalPorchSF' in df.columns:
            df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int)

        self.df = df
        print("✓ Boolean features created")
        return df

    # PIPELINE

    def run_full_pipeline(self, create_segments=True, cap_features=True):
        """
        Run complete data processing pipeline

        Args:
            create_segments: If True, creates market segments for separate modeling
            cap_features: If True, creates capped versions of key features
        """
        print("\n" + "=" * 60)
        print("RUNNING FULL DATA PROCESSING PIPELINE")
        print("=" * 60)

        # 1. Missing value analysis and treatment
        print("\n[1/9] Analyzing missing values...")
        self.analyze_missing_values()
        self.handle_missing_values()

        # 2. Outlier detection
        print("\n[2/9] Detecting outliers...")
        self.detect_outliers()
        self.detect_domain_outliers()

        # 3. Feature engineering
        print("\n[3/9] Creating size features...")
        self.create_size_features()

        print("\n[4/9] Creating age features...")
        self.create_age_features()

        print("\n[5/9] Creating room features...")
        self.create_room_features()

        print("\n[6/9] Creating quality features...")
        self.create_quality_features()

        print("\n[7/9] Creating temporal features...")
        self.create_temporal_features()
        self.create_boolean_features()

        # 4. Capping approach (keep original + capped versions)
        if cap_features:
            print("\n[8/9] Creating capped feature versions...")
            features_to_cap = ['GrLivArea', 'LotArea', 'TotalBsmtSF',
                              '1stFlrSF', 'GarageArea', 'TotalSF']
            available_to_cap = [f for f in features_to_cap if f in self.df.columns]
            if available_to_cap:
                self.cap_outliers(available_to_cap, percentile=99, keep_original=True)

        # 5. Market segmentation
        if create_segments and 'SalePrice' in self.df.columns:
            print("\n[9/9] Creating market segments...")
            self.create_market_segments(price_col='SalePrice',
                                       thresholds=[500000, 1000000])

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Total features: {len(self.df.columns)}")

        if hasattr(self, 'segments'):
            print(f"\n✓ Market segments created:")
            print(f"  - Mainstream: {len(self.segments['mainstream'])} homes")
            print(f"  - Upper: {len(self.segments['upper'])} homes")
            print(f"  - Luxury: {len(self.segments['luxury'])} homes")

        return self.df

    def get_segment_data(self, segment='mainstream'):
        """
        Get data for a specific market segment

        Args:
            segment: 'mainstream', 'upper', 'luxury', or 'full'

        Returns:
            DataFrame for the specified segment
        """
        if not hasattr(self, 'segments'):
            print("Error: Run create_market_segments() first")
            return None

        if segment not in self.segments:
            print(f"Error: Invalid segment. Choose from: {list(self.segments.keys())}")
            return None

        return self.segments[segment]
