import data_retrieval as dr
import nebula_image_downloader as nid
import negative_image_downloader as nidr
import dataset_exporter as de
import model_training as mt

dr.retrieve_dataset()
images = nid.load_and_preprocess_images_parallel('StDr.csv')
nid.clear_astropy_cache()

not_images = nidr.download_negative_samples()
nidr.clear_astropy_cache()

de.export_dataset(images, not_images)

history = mt.run_training(tune_hyperparams=False)
