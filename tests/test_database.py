import psycopg2

import DB.database_manager as db_module
from DB.database_manager import DatabaseManager


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        if self.connection.error is not None:
            raise self.connection.error

    def fetchall(self):
        return [(1, "row")]


class FakeConnection:
    def __init__(self):
        self.closed = 0
        self.error = None
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def make_manager(monkeypatch):
    connections = []

    def fake_connect(**kwargs):
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(db_module.psycopg2, "connect", fake_connect)
    manager = DatabaseManager("db", "user", "pw", "127.0.0.1", "5432")
    return manager, connections


def test_connect_is_idempotent(monkeypatch):
    manager, connections = make_manager(monkeypatch)

    assert manager.connect() and manager.connect()

    assert len(connections) == 1  # раньше второй connect() терял первое соединение


def test_read_query_closes_transaction(monkeypatch):
    manager, connections = make_manager(monkeypatch)
    manager.connect()

    assert manager.execute_read_query("SELECT 1") == [(1, "row")]
    assert connections[0].commits == 1


def test_any_database_error_rolls_back(monkeypatch):
    manager, connections = make_manager(monkeypatch)
    manager.connect()
    # раньше ловился только OperationalError, остальные ошибки пробивали вверх
    connections[0].error = psycopg2.ProgrammingError("relation does not exist")

    assert manager.execute_read_query("SELECT * FROM nope") == []
    assert manager.execute_query("DELETE FROM nope") is False
    assert connections[0].rollbacks == 2
