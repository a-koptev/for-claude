import logging
from typing import List, Dict, Tuple

import numpy as np
from psycopg2 import Error as PsycopgError
from psycopg2.extras import execute_batch

from DB.database_manager import DatabaseManager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarFeatureRepository:
    """Репозиторий для работы с признаками автомобилей"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.db.connect()

    @staticmethod
    def _vector_to_postgres_array(vector: np.ndarray) -> str:
        """Конвертирует numpy массив в строку формата PostgreSQL массива"""
        return '[' + ','.join(map(str, vector)) + ']'

    @staticmethod
    def _postgres_array_to_vector(array_str: str) -> np.ndarray:
        """Конвертирует строку PostgreSQL массива в numpy массив"""
        return np.fromstring(array_str[1:-1], sep=',')

    def add_single_feature(self, car_id: int, feature_vector: np.ndarray) -> bool:
        """Добавляет один вектор признаков для автомобиля"""
        query = """
            INSERT INTO features (car_id, feature)
            VALUES (%s, %s)
        """
        feature_str = self._vector_to_postgres_array(feature_vector)
        return self.db.execute_query(query, (car_id, feature_str))

    def add_batch_features(self, car_id: int, feature_vectors: List[np.ndarray]) -> bool:
        """Добавляет пакет векторов признаков для автомобиля"""
        query = """
            INSERT INTO features (car_id, feature)
            VALUES (%s, %s)
        """

        data_to_insert = [
            (car_id, self._vector_to_postgres_array(vector))
            for vector in feature_vectors
        ]

        if not data_to_insert:
            logger.warning("Нет данных для вставки")
            return False

        try:
            with self.db.connection.cursor() as cursor:
                execute_batch(cursor, query, data_to_insert)
                self.db.connection.commit()
                logger.info(f"Успешно добавлено {len(feature_vectors)} векторов для car_id {car_id}")
                return True
        except PsycopgError as e:
            logger.error(f"Ошибка пакетной вставки: {e}")
            self.db.connection.rollback()
            return False

    def get_all_features(self) -> List[Tuple[int, np.ndarray]]:
        """Возвращает все векторы признаков из базы данных"""
        query = "SELECT car_id, feature FROM features"
        results = self.db.execute_read_query(query)

        cars = []
        for car_id, feature_str in results:
            try:
                feature_vector = self._postgres_array_to_vector(feature_str)
                cars.append((car_id, feature_vector))
            except ValueError as e:
                logger.error(f"Ошибка преобразования вектора для car_id {car_id}: {e}")

        return cars

    def get_features_by_car_ids(self, car_ids: List[int]) -> Dict[int, np.ndarray]:
        """Возвращает векторы признаков для указанных car_id"""
        if not car_ids:
            return {}

        query = """
            SELECT car_id, feature
            FROM features 
            WHERE car_id IN %s
        """
        results = self.db.execute_read_query(query, (tuple(car_ids),))

        features_dict = {}
        for car_id, feature_str in results:
            try:
                feature_vector = self._postgres_array_to_vector(feature_str)
                if car_id not in features_dict:
                    features_dict[car_id] = []
                features_dict[car_id].append(feature_vector)
            except ValueError as e:
                logger.error(f"Ошибка преобразования вектора для car_id {car_id}: {e}")

        # Конвертируем списки в numpy массивы
        return {
            car_id: np.array(vectors)
            for car_id, vectors in features_dict.items()
        }
