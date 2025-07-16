parquet_paths_one_file = ['/home/hsp252/Development/intrusion_sorted/DDoS/intrusion_normal_DDoS_10_final.parquet',
                          '/home/hsp252/Development/intrusion_sorted/DoS/intrusion_normal_DoS_1_final.parquet',
                          '/home/hsp252/Development/intrusion_sorted/injection/intrusion_injection_normal1_final.parquet',
                          '/home/hsp252/Development/intrusion_sorted/mitm/intrusion_MITM_normal1_final.parquet',
                          '/home/hsp252/Development/intrusion_sorted/runsomware/intrusion_normal_runsomware1_final.parquet',
                          '/home/hsp252/Development/intrusion_sorted/scanning/intrusion_normal_scanning1_final.parquet',
                          '/home/hsp252/Development/intrusion_sorted/backdoor/intrusion_normal_backdoor_final.parquet',
                          '/home/hsp252/Development/benign/benign_normal_10_final.parquet']

parquet_paths = ['/home/hsp252/Development/intrusion_sorted/DDoS',
                 '/home/hsp252/Development/intrusion_sorted/DoS',
                 '/home/hsp252/Development/intrusion_sorted/injection',
                 '/home/hsp252/Development/intrusion_sorted/mitm',
                 '/home/hsp252/Development/intrusion_sorted/runsomware',
                 '/home/hsp252/Development/intrusion_sorted/scanning',
                 '/home/hsp252/Development/intrusion_sorted/backdoor',
                 '/home/hsp252/Development/benign']

s1_features = [
    "gq_mean_queue_length", "gq_median_queue_length", "gq_mean_growth_rate",
    "gq_median_growth_rate", "gq_growth_rates_percentage", "gq_growth_rates",
    "gq_popped_pkts_iteration", "gq_popped_pkts_cumulative", "gq_shortest_queue",
    "gq_longest_queue", "gq_host_count", "gq_host_count_diff",
    "gq_mean_host_queue_length", "gq_median_host_queue_length",
    "gq_host_queue_length_variance",
    "gq_host_queue_length_entropy", "gq_mean_protocol_queue_length",
    "gq_protocol_queue_length_variance", "gq_protocol_queue_length_entropy",
    "gq_protocol_queue_count", "gq_protocol_queue_count_diff",
    "gq_median_protocol_queue_length", "gq_protocol_queue_count_layer_4_below",
    "gq_total_cache_len", "gq_active_router_count", "gq_active_router_in_cache_count",
    "gq_non_protocol_non_host_queues",
    "gq_mean_priority", "gq_median_priority",'gq_cq_connection_count_total',
    'gq_cq_connection_count_total_diff', 'gq_cq_connection_queues_length_mean',
    'gq_cq_connection_queues_length_median', 'gq_cq_connection_queues_length_sum_iteration',
    'gq_protocol_queue_distribution_overall_percent', 'gq_protocol_distribution_l4_below_percent',
    'gq_protocol_distribution_l5_plus_percent', 'gq_protocol_queue_count_layer_5_plus',
    'gq_highest_priority', 'gq_lowest_priority'
]

s2_features = [
    "sq_identifier", "sq_queue_length", "sq_priority", "sq_popped_pkts",
    "sq_most_active_protocols", "time_window_start", "time_window_end",
    "growth_rate_percentage", "growth_rate", "is_longest_queue",
    "is_shortest_queue", "is_most_active_host_based_on_packets",
    'sq_host_or_protocol_queue',
    'sq_relative_connection_count',
    'sq_relative_connection_queue_length',
    'sq_relative_mean_conn_queue_length',
    'sq_relative_median_conn_queue_length',
    'sq_relative_priority',
    'sq_relative_protocol_queue_length',
    'sq_relative_queue_length',
    'sq_relative_host_queue_length',
]

s3_features = [
    "sq_identifier", "sq_connection_count", "sq_connection_count_diff",
    "sq_connection_type_counts", "sq_cq_connection_queues_length_sum_iteration",
    "sq_cq_connection_queues_length_median", "sq_cq_connection_queues_length_mean",
    "time_window_start", "time_window_end", "connection_tcp_udp_other_ratio"
]

connection_features = [
    "selected_queue", "window_time_key", "analysis_time_key", "host_ct_dst_addresses",
    "host_mode_dst_addresses", "host_ct_src_ports", "host_mode_src_ports",
    "host_ct_dst_ports", "host_mode_dst_ports", "host_ct_protocols",
    "host_mode_protocols", "host_ct_syn", "host_ct_ack", "host_ct_fin",
    "host_ct_cwr", "host_ct_psh", "host_ct_urg", "host_ct_ecn", "host_ct_rst",
    "host_mode_tcp_flags", "host_ct_pkt_lens", "host_mode_pkt_lens",
    "conn_window_start_time", "conn_window_end_time", "conn_protocol",
    "conn_src_ip", "conn_dst_ip", "conn_src_port", "conn_dst_port",
    "conn_connection_state", "conn_duration", "conn_packets/s", "conn_bytes_src",
    "conn_bytes_dst", "conn_src_ttl", "conn_dst_ttl", "conn_#src_pkts",
    "conn_#dst_pkts", "conn_src_dst_ratio", "conn_min_src_pkt_len",
    "conn_max_src_pkt_len", "conn_mean_src_pkt_len", "conn_stdev_src_pkt_len",
    "conn_mode_src_pkt_len", "conn_median_src_pkt_len", "conn_min_dst_pkt_len",
    "conn_max_dst_pkt_len", "conn_mean_dst_pkt_len", "conn_stdev_dst_pkt_len",
    "conn_mode_dst_pkt_len", "conn_median_dst_pkt_len", "conn_min_src_iats",
    "conn_max_src_iats", "conn_mean_src_iats", "conn_stdev_src_iats",
    "conn_median_src_iats", "conn_var_src_iats", "conn_min_dst_iats",
    "conn_max_dst_iats", "conn_mean_dst_iats", "conn_stdev_dst_iats",
    "conn_median_dst_iats", "conn_var_dst_iats", "conn_ct_src_syn",
    "conn_ct_src_ack", "conn_ct_src_fin", "conn_ct_src_cwr", "conn_ct_dst_syn",
    "conn_ct_dst_ack", "conn_ct_dst_fin", "conn_ct_dst_cwr",
    "conn_min_src_payload_len", "conn_max_src_payload_len",
    "conn_stdev_src_payload_len", "conn_median_src_payload_len",
    "conn_var_src_payload_len", "conn_mean_src_payload_len",
    "conn_1st_quartile_src_payload_len", "conn_3rd_quartile_src_payload_len",
    "conn_min_max_diff_src_payload_len", "conn_rms_src_payload_len",
    "conn_g1_skew_src_payload_len", "conn_G1_skew_src_payload_len",
    "conn_sk1_skew_src_payload_len", "conn_sk2_skew_src_payload_len",
    "conn_galton_skew_src_payload_len", "conn_entropy_src_payload_len",
    "conn_kurtosis_src_payload_len", "conn_coeff_variation_src_payload_len",
    "conn_min_dst_payload_len", "conn_max_dst_payload_len",
    "conn_stdev_dst_payload_len", "conn_median_dst_payload_len",
    "conn_var_dst_payload_len", "conn_mean_dst_payload_len",
    "conn_1st_quartile_dst_payload_len", "conn_3rd_quartile_dst_payload_len",
    "conn_min_max_diff_dst_payload_len", "conn_rms_dst_payload_len",
    "conn_g1_skew_dst_payload_len", "conn_G1_skew_dst_payload_len",
    "conn_sk1_skew_dst_payload_len", "conn_sk2_skew_dst_payload_len",
    "conn_galton_skew_dst_payload_len", "conn_entropy_dst_payload_len",
    "conn_kurtosis_dst_payload_len", "conn_coeff_variation_dst_payload_len",
    "conn_mean_relative_times", "conn_median_relative_times",
    "conn_1st_quartile_relative_times", "conn_3rd_quartile_relative_times",
    "conn_significant_spaces_src", "conn_significant_spaces_dst",
    "conn_count_of_zeros_src", "conn_count_of_zeros_dst",
    'conn_G1_skew_dst_payload_len_1',
    'conn_G1_skew_src_payload_len_1',
]

attack_types = ['ddos', 'dos', 'injection', 'mitm', 'runsomware', 'scanning', 'backdoor', 'normal']

NR_MIN_ELEMENTS_IN_ONE_FILE = 27278 # -> nicht viel
NR_MAX_ELEMENTS_IN_ONE_FILE = 6083416

NR_MAX_ELEMENTS_IN_ALL_FILES = 75900167
NR_MIN_ELEMENTS_IN_ALL_FILES = 29243 # -> nicht viel

NR_NORMAL_SAMPLES = 796530