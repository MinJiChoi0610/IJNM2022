import ni_mon_client
from datetime import datetime, timedelta
from config import cfg
from ni_mon_client.rest import ApiException


ni_mon_client_cfg = ni_mon_client.Configuration()
ni_mon_client_cfg.host = cfg["ni_mon"]["host"]

ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

vnf_id = "64d0565a-addb-48ee-ade0-0e5e4dcfd110"
measurement_type = "cpu_usage___value___gauge"


def test_measurement_vnf():
    start_time = datetime.now() - timedelta(minutes=5)
    end_time = datetime.now()

    measurement_results = ni_mon_api.get_measurement(vnf_id, measurement_type, start_time, end_time)
    print(measurement_results)

test_measurement_vnf()