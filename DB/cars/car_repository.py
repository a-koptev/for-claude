import logging
from typing import Optional

from psycopg2 import Error as PsycopgError

from DB.database_manager import DatabaseManager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarRepository:
    """Репозиторий для работы с автомобилями"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_car(self, car_number: str) -> Optional[int]:
        """Добавляет новый автомобиль и возвращает его ID"""
        query = """
            INSERT INTO cars (car_number)
            VALUES (%s)
            RETURNING car_id
        """

        try:
            with self.db.connection.cursor() as cursor:
                cursor.execute(query, (car_number,))
                car_id = cursor.fetchone()[0]
                self.db.connection.commit()
                logger.info(f"Успешно добавлен автомобиль {car_number} с ID {car_id}")
                return car_id
        except PsycopgError as e:
            logger.error(f"Ошибка добавления автомобиля: {e}")
            self.db.connection.rollback()
            return None
