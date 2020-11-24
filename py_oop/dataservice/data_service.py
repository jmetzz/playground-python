import copy
import glob
import logging
import os
import time
from abc import ABC
from concurrent import futures
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

from py_oop.dataservice import config
from py_oop.dataservice.db import get_data_frame
from py_oop.dataservice.query import (
    get_market_config,
    get_last_period_seq_per_cell,
    get_all_market_config,
    get_all_data,
)
from py_oop.dataservice.utils import (
    timed,
    audited,
    get_current_db_id,
    get_previous_week_no,
)

logger = logging.getLogger(__name__)

DATA_CACHE = {}


class AbstractDataService(ABC):
    MARKET_CONFIG_COLUMNS = [
        "country_code",
        "group_code",
        "low_price_percentage",
        "high_price_percentage",
        "medium_price_percentage",
        "lower_price_range_threshold",
        "upper_price_range_threshold",
    ]

    DATA_COLUMNS = [
        "item_id",
        "group_code",
        "country_code",
        "period_seq",
        "my_rank",
        "competitor_item_id",
        "loc_distance_euclidean",
        "distance_euclidean",
        "distribution_overlap",
        "brand",
        "price_own",
        "loc_price_own",
        "salesunits_own",
        "loc_salesunits_own",
        "sufficient_sales_own",
        "wgt_distr_own",
        "loc_wgt_distr_own",
        "brand_competitor",
        "price_competitor",
        "loc_price_competitor",
        "salesunits_competitor",
        "loc_salesunits_competitor",
        "wgt_distr_competitor",
        "loc_wgt_distr_competitor",
        "no_of_periods_in_focus",
        "tpr_efficiency_own",
        "loc_tpr_efficiency_own",
    ]

    def get_data(self, group_code: str, country_code: str, period_seq: int) -> pd.DataFrame:
        ...

    def get_market_configuration(self, group_code: str, country_code: str) -> pd.DataFrame:
        ...

    def startup(self, db_id: Optional[int] = None):
        ...

    def list_cells(self) -> List[Tuple[str, str, str]]:
        ...


class CacheDataService(AbstractDataService):
    """
    A Data service that loads its data at startup time, and stores it in memory
    """

    @audited
    def get_data(self, group_code: str, country_code: str, period_seq: int) -> pd.DataFrame:
        """
        Grab market data from in-memory (cache.CELLS)
        """
        cell_name: str = f"{group_code}-{country_code}"

        if cell_name not in DATA_CACHE:
            logger.info("MARKET_NOT_EXISTS", extra={"description": cell_name})
            return pd.DataFrame(columns=self.DATA_COLUMNS)

        return copy.deepcopy(DATA_CACHE[cell_name]["data"])

    def get_market_configuration(self, group_code: str, country_code: str) -> pd.DataFrame:
        """
        Grab market config from in-memory (cache.CELLS)
        """
        cell_name: str = f"{group_code}-{country_code}"

        if cell_name not in DATA_CACHE:
            logger.info("MARKET_NOT_EXISTS", extra={"description": cell_name})
            return pd.DataFrame(columns=self.DATA_COLUMNS)

        return copy.deepcopy(DATA_CACHE[cell_name]["config"])

    def startup(self, db_id: Optional[int] = None):
        """
        Populate our in-memory cache (Not shared)
        """
        start = time.perf_counter()
        last_periods = get_previous_week_no()

        executor = ProcessPoolExecutor if config.EXECUTOR == "processes" else ThreadPoolExecutor

        market_configuration = get_data_frame(get_all_market_config)
        total = len(market_configuration)
        logger.info(
            "LOADING_CACHE",
            extra={"count": len(market_configuration), "executor": config.EXECUTOR, "backend": config.DATA_SERVICE,},
        )

        with executor() as executor_instance:
            tasks = [
                executor_instance.submit(self._update_cell, row.item_group_code, row.country_code, last_periods,)
                for row in market_configuration.itertuples()
            ]

            for ix, future in enumerate(futures.as_completed(tasks)):
                logger.info("CACHE_PROGRESS", extra={"current": ix, "total": total})
                cell_name, cell_data = future.result()

                # avoid caching empty data
                if cell_data:
                    DATA_CACHE[cell_name] = cell_data

        logger.info("CACHE_LOADED_OK", extra={"elapsed": time.perf_counter() - start})

    def _update_cell(
        self, group_code: str, country_code: str, periods: Dict[Tuple[str, str], int],
    ) -> Tuple[str, Dict[str, Any]]:
        """Executes a query for a specific cell, returns a tuple of cell name and data.

        Args:
            group_code:
            country_code:
            periods: a dictionary mapping each cell to the relevant period_seq that should be fetched
        """
        start = time.perf_counter()
        cell_name: str = f"{group_code}-{country_code}"
        cell_data: Dict[str, Any] = {}

        if (group_code, country_code) not in periods:
            logger.error(
                "CELL_LOADED_KEY_ERROR", extra={"cell_name": cell_name, "elapsed": time.perf_counter() - start},
            )
            return cell_name, cell_data

        try:
            period_seq: int = periods[(group_code, country_code)]
            data_params: Dict = {
                "group_code": group_code,
                "country_code": country_code,
                "period_seq": period_seq,
            }

            market_config = get_data_frame(get_market_config, {"group_code": group_code, "country_code": country_code},)

            logger.info("CELL_LOAD_START", extra={"cell": cell_name})

            data = get_data_frame(get_all_data, data_params)
            cell_data = {
                "data": data,
                "period_seq": period_seq,
                "config": market_config,
            }

            logger.info(
                "CELL_LOADED_OK",
                extra={
                    "cell_name": cell_name,
                    "period_seq": period_seq,
                    "shape": data.shape,
                    "elapsed": time.perf_counter() - start,
                },
            )

        except Exception as e:
            logger.error(
                "CELL_LOADED_FAILED",
                extra={"cell_name": cell_name, "description": str(e), "elapsed": time.perf_counter() - start,},
            )
            raise

        return cell_name, cell_data

    def list_cells(self) -> List[Tuple[str, str, str]]:
        cells = []
        for key, data in DATA_CACHE.items():
            if "-" not in key:
                continue
            cell, country = key.split("-")
            period = str(data["data"]["period_seq"].iloc[0])
            cells.append((cell, country, period))

        return cells


class TableDataService(AbstractDataService):
    """
    A data service that goes straight into the database to perform queries
    """

    @audited
    def get_data(self, group_code: str, country_code: str, period_seq: int) -> pd.DataFrame:
        """
        Grab market data from the database
        """
        data = get_data_frame(
            get_all_data, {"group_code": group_code, "country_code": country_code, "period_seq": period_seq,},
        )
        return data

    def get_market_configuration(self, group_code: str, country_code: str) -> pd.DataFrame:
        """
        Grab market config from the database
        """
        market_config = get_data_frame(get_market_config, {"group_code": group_code, "country_code": country_code},)
        if market_config.empty:
            return pd.DataFrame(columns=self.MARKET_CONFIG_COLUMNS)
        return market_config

    def startup(self, db_id: Optional[int] = None):
        """
        No cache preperation needed
        """
        pass

    def list_cells(self) -> List[Tuple[str, str, str]]:
        period_data = get_data_frame(get_last_period_seq_per_cell)
        market_config = get_data_frame(get_all_market_config)
        iter_cells = market_config.merge(period_data, on=["group_code", "country_code"])
        iter_cells["period_seq"] = iter_cells["period_seq"].astype(str)
        groups = iter_cells.groupby(["group_code", "country_code", "period_seq"]).groups
        return list(groups)


class FileSystemDataService(AbstractDataService):
    """
    A data service that populates a filesystem cache offline, and access pre-computed results
    """

    @audited
    def get_data(self, item_group_code: str, country_code: str, period_seq: int, db_id: int = 0) -> pd.DataFrame:
        """
        Grab market data from our FS cache
        """
        filename = os.path.join(
            "/tmp", "ce_cache", str(db_id), item_group_code, country_code, str(period_seq), "input.parquet",
        )
        if not os.path.exists(filename):
            return pd.DataFrame(columns=self.DATA_COLUMNS)

        return pd.read_parquet(filename)

    def get_market_configuration(self, item_group_code: str, country_code: str, db_id: int = 0) -> pd.DataFrame:
        """
        Grab market config from our FS cache
        """
        filename = os.path.join("/tmp", "ce_cache", str(db_id), "market_config.parquet")
        if not os.path.exists(filename):
            return pd.DataFrame(columns=self.DATA_COLUMNS)

        df = pd.read_parquet(filename)

        if item_group_code:
            df = df.query(f"item_group_code == '{item_group_code}'")

        if country_code:
            df = df.query(f"country_code == '{country_code}'")

        return df

    def get_moc_thresholds(
        self, group: Optional[str] = None, country: Optional[str] = None, period: Optional[int] = None, db_id: int = 0,
    ) -> pd.DataFrame:

        filename = os.path.join("/tmp", "ce_cache", str(db_id), "thresholds.parquet")
        if not os.path.exists(filename):
            return pd.DataFrame(columns=self.THRESHOLDS_COLUMNS)

        df = pd.read_parquet(filename)
        if group:
            df = df.query(f"item_group_code == '{group}'")

        if country:
            df = df.query(f"country_code == '{country}'")

        return df

    def startup(self, db_id: Optional[int] = None):
        """
        Prime the cache if we are running locally, otherwise, let liveness.py do it offline
        """
        df = pd.DataFrame(self.list_cells(), columns=["item_group_code", "country_code", "period_seq"])
        self._populate_fs_cache(current_cells_df=df)

    def list_cells(self) -> List[Tuple[str, str, str]]:
        cache_dir = os.path.join("/tmp", "ce_cache")
        cells = []
        for file in glob.glob(cache_dir + "/**/*.parquet", recursive=True):
            parts = file.split("/")
            if len(parts) == 8:
                cell, country, period = parts[4:7]
                cells.append((cell, country, period))

        return cells

    @timed
    def _populate_fs_cache(
        self,
        db_id: Optional[int] = None,
        force: Optional[bool] = False,
        current_cells_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Get the data that we need for running the algo online, and store it in the filesystem

        Produces the following files:

        /tmp/ce_cache/<db_id>/thresholds.parquet
        /tmp/ce_cache/<db_id>/market_config.parquet
        /tmp/ce_cache/<db_id>/<item_group>/<country>/<period>/input.parquet (Per cell)

        """
        db_id = db_id or get_current_db_id()
        cache_dir = os.path.join("/tmp", "ce_cache", str(db_id))

        executor = ProcessPoolExecutor if config.EXECUTOR == "processes" else ThreadPoolExecutor

        #################
        # Market config #
        #################
        market_config = get_data_frame(get_all_market_config)
        filename = os.path.join("/tmp", "ce_cache", str(db_id), "market_config.parquet")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not os.path.exists(filename) or force:
            with open(filename, "wb+") as fp:
                market_config.to_parquet(fp, compression="gzip", engine="pyarrow", index=False)

        #############
        # Cell data #
        #############
        logger.info(
            "LOADING_CACHE",
            extra={"count": len(market_config), "executor": config.EXECUTOR, "backend": config.DATA_SERVICE,},
        )
        period_data = get_data_frame(get_last_period_seq_per_cell)
        iter_cells = market_config.merge(period_data, on=["group_code", "country_code"]).copy()

        # Filtering by existing cache keys
        if not force and current_cells_df is not None:
            hash_cols = ["group_code", "country_code", "period_seq"]
            iter_cells["hash"] = pd.util.hash_pandas_object(iter_cells[hash_cols], index=False)
            current_cells_df["hash"] = pd.util.hash_pandas_object(current_cells_df[hash_cols], index=False)
            iter_cells = iter_cells.loc[~iter_cells["hash"].isin(current_cells_df["hash"])]

        if iter_cells.empty:
            logger.error("MARKET_CONFIG_EMPTY")
            return

        with executor() as executor_instance:
            for row in iter_cells.itertuples():
                params = {
                    "group_code": row.group_code,
                    "country_code": row.country_code,
                    "period_seq": row.period_seq,
                }
                filename = os.path.join(
                    cache_dir, row.group_code, row.country_code, str(row.period_seq), "input.parquet",
                )
                executor_instance.submit(self._cache_results, get_all_data, params, filename)

        logger.info("CACHE_LOADED_OK")

    def _cache_results(self, query: str, params: Dict[str, Any], filename: str) -> None:
        """
        go get the data we need for the given cell
        """

        logger.info("CELL_LOAD_START", extra=params)

        df = get_data_frame(query, params)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb+") as fp:
            df.to_parquet(fp, compression="gzip", engine="pyarrow", index=False)

        logger.info("CELL_LOAD_END", extra=params)

    def list_cells(self) -> List[Tuple[str, str, str]]:
        cache_dir = os.path.join("/tmp", "ce_cache")
        cells = []
        for file in glob.glob(cache_dir + "/**/*.parquet", recursive=True):
            parts = file.split("/")
            if len(parts) == 8:
                cell, country, period = parts[4:7]
                cells.append((cell, country, period))

        return cells


def get_data_service() -> AbstractDataService:
    return {"memory": CacheDataService(), "table": TableDataService(), "filesystem": FileSystemDataService(),}.get(
        str(config.DATA_SERVICE), CacheDataService()
    )
