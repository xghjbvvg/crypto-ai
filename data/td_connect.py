import taos

conn = taos.connect(
    host="172.17.18.158",  # 你 TDengine 服务器的 IP
    port=6030,
    user="root",
    password="taosdata",
    database="cryptocompare_db"
)

cursor = conn.cursor()

# sql = f"SELECT ts, open, high, low, close, volumefrom AS volume FROM ada_usdt_kline ORDER BY ts DESC LIMIT 100"
# cursor.execute(sql)
# rows = cursor.fetchall()
# sql = """
# INSERT INTO btc_usdt _kline
# USING kline 
# TAGS ('BTC', 'USDT') 
# VALUES ('2025-08-31 00:00:00', 27000.0, 27500.0, 26800.0, 27300.0, 100.5, 2730000.0);
# """

# cursor.execute(sql)
# conn.close()