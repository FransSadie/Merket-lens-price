from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="Market Lens", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(default="sqlite:///./market_lens.db", alias="DATABASE_URL")

    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    news_api_base_url: str = Field(default="https://newsapi.org/v2", alias="NEWS_API_BASE_URL")
    news_query: str = Field(
        default="(stock OR market OR earnings OR inflation OR fed)", alias="NEWS_QUERY"
    )
    news_language: str = Field(default="en", alias="NEWS_LANGUAGE")
    news_page_size: int = Field(default=50, alias="NEWS_PAGE_SIZE")
    news_max_pages: int = Field(default=3, alias="NEWS_MAX_PAGES")
    news_lookback_days: int = Field(default=7, alias="NEWS_LOOKBACK_DAYS")
    historical_news_dir: str = Field(default="./data/historical_news", alias="HISTORICAL_NEWS_DIR")
    historical_news_format: str = Field(default="csv", alias="HISTORICAL_NEWS_FORMAT")
    historical_news_batch_limit: int = Field(default=0, alias="HISTORICAL_NEWS_BATCH_LIMIT")
    gdelt_gkg_raw_dir: str = Field(default="./data/gdelt_gkg_raw", alias="GDELT_GKG_RAW_DIR")
    gdelt_gkg_normalized_dir: str = Field(default="./data/historical_news", alias="GDELT_GKG_NORMALIZED_DIR")
    gdelt_gkg_batch_limit: int = Field(default=0, alias="GDELT_GKG_BATCH_LIMIT")
    gdelt_gkg_row_limit: int = Field(default=0, alias="GDELT_GKG_ROW_LIMIT")

    market_tickers: str = Field(default="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,AVGO,CRM,ORCL,ADBE,NFLX,CSCO,QCOM,JPM,GS,MS,V,MA,UNH,LLY,COST,WMT,KO,PEP,XOM,CVX,CAT,BA,GE,SPY,QQQ,IWM,DIA,XLF,XLK,XLV,XLY,XLI,XLE,XLB,XLU,XLP,TLT,GLD", alias="MARKET_TICKERS")
    market_history_period: str = Field(default="1y", alias="MARKET_HISTORY_PERIOD")
    market_history_interval: str = Field(default="1d", alias="MARKET_HISTORY_INTERVAL")
    ingest_interval_minutes: int = Field(default=15, alias="INGEST_INTERVAL_MINUTES")
    model_artifacts_dir: str = Field(default="./artifacts", alias="MODEL_ARTIFACTS_DIR")
    model_type: str = Field(default="lightgbm", alias="MODEL_TYPE")
    predict_hold_threshold: float = Field(default=0.6, alias="PREDICT_HOLD_THRESHOLD")
    enable_news_pipeline: bool = Field(default=False, alias="ENABLE_NEWS_PIPELINE")
    training_feature_set: str = Field(default="price_only", alias="TRAINING_FEATURE_SET")
    training_horizon_days: int = Field(default=5, alias="TRAINING_HORIZON_DAYS")
    training_target_return_threshold: float = Field(default=0.004, alias="TRAINING_TARGET_RETURN_THRESHOLD")

    lightgbm_num_leaves: int = Field(default=15, alias="LIGHTGBM_NUM_LEAVES")
    lightgbm_learning_rate: float = Field(default=0.05, alias="LIGHTGBM_LEARNING_RATE")
    lightgbm_n_estimators: int = Field(default=200, alias="LIGHTGBM_N_ESTIMATORS")
    lightgbm_max_depth: int = Field(default=6, alias="LIGHTGBM_MAX_DEPTH")
    lightgbm_min_child_samples: int = Field(default=20, alias="LIGHTGBM_MIN_CHILD_SAMPLES")
    lightgbm_subsample: float = Field(default=0.9, alias="LIGHTGBM_SUBSAMPLE")
    lightgbm_colsample_bytree: float = Field(default=0.9, alias="LIGHTGBM_COLSAMPLE_BYTREE")

    @property
    def market_ticker_list(self) -> List[str]:
        return [ticker.strip().upper() for ticker in self.market_tickers.split(",") if ticker.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
