from helper_functions import all_sensor_time_plot_separate,all_sensor_fft_plot_separate,single_sensor_fft_plot,every_defect_mode_harmonics_plot


path = 'Balanced_data'
random_path = 'random_data'
all_sensor_time_plot_separate(path,10)
all_sensor_fft_plot_separate(path,10)
single_sensor_fft_plot(random_path,10)
every_defect_mode_harmonics_plot(path,0,1,54,58)