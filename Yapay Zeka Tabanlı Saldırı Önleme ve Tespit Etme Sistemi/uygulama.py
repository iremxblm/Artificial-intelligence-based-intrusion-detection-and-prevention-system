from scapy.all import sniff
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import ipaddress
import time

# Kaydedilmiş modeli yükle
model = joblib.load("trained_decision_tree_model.joblib")

# Yakalanan paketleri depolamak için veri çerçevesi oluştur
column_names = ['Source_IP']
packet_data = pd.DataFrame(columns=column_names)

# Başlangıç zamanını al
start_time = time.time()

# Paket sayısı ve zamanlayıcı değerlerini tanımla
total_packets = 0
elapsed_seconds = 1

def handle_packet(pkt):
    global packet_data, total_packets, elapsed_seconds

    if pkt.haslayer('IP'):
        src_ip = pkt['IP'].src

        # IP adresini sayısal değere dönüştür
        numeric_ip = int(ipaddress.ip_address(src_ip))

        # Yakalanan paketi veri çerçevesine ekle
        packet_data.loc[len(packet_data)] = [numeric_ip]

        total_packets += 1

        # Eğer paket sayısı eşiği aşıyorsa saldırıyı tespit et
        if total_packets > 100:
            # Ölçeklendirme
            scaler = StandardScaler()
            scaled_packets = scaler.fit_transform(packet_data)

            # Saldırıyı tespit et
            prediction = model.predict(scaled_packets)

            # Eğer saldırı tespit edilirse uyarı ver
            if any(prediction):
                print("Potansiyel saldırı tespit edildi!")

            # Veri çerçevesini sıfırla
            packet_data.drop(packet_data.index, inplace=True)
            total_packets = 0

    # Her saniyede bir sonuçları yazdır
    current_time = time.time()
    time_diff = current_time - start_time
    if time_diff >= elapsed_seconds:
        print(f"Minute {elapsed_seconds}:")
        for ip, count in packet_data['Source_IP'].value_counts().items():
            print(f"{ipaddress.ip_address(ip)}: {count} paket")
        print("----------------------")
        elapsed_seconds += 1
        if elapsed_seconds > time_diff:
            elapsed_seconds = int(time_diff) + 1

# Paket yakalama işlemine başla
sniff(prn=handle_packet, count=0, filter="tcp and (port 80 or port 443)")

