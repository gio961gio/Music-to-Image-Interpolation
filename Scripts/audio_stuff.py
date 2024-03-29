import os
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf
import shutil
from pydub import AudioSegment
import torch

class Audio_stuff:
    def __init__ (self, num_segments):
        self.num_segments = num_segments+1
        
        #Caricamento file audio
        audio_length_ms, valore_pad, sample_rate,audio_data,input_audio_path = self.load_padCalculator()
        
        # Specifica la lunghezza target desiderata in millisecondi
        target_length_ms = audio_length_ms + valore_pad  # Aggiungi pad in millisecondi

        # Calcola la lunghezza target in campioni
        target_length_samples = int((target_length_ms / 1000) * sample_rate)

        # Esegui il padding dell'audio
        padded_audio = self.pad_audio(audio_data, target_length_samples)

        # Salva il file audio paddato nella stessa cartella del file originale
        padded_file_path = os.path.splitext(input_audio_path)[0] + "_padded.wav"
        sf.write(padded_file_path, padded_audio, sample_rate)

        # Rimuovi il file originale
        os.remove(input_audio_path)


        self.input_audio_path = padded_file_path

        # Specifica il percorso della cartella da cancellare
        folder_path = "/content/audio_segments"

        # Verifica se la cartella esiste prima di cancellarla
        if os.path.exists(folder_path):
        # Cancella la cartella e tutti i suoi contenuti
            shutil.rmtree(folder_path)
            del folder_path

        output_folder = "/content/audio_segments"
        self.split_audio(self.input_audio_path, output_folder, self.num_segments)


        # Specifica il percorso della cartella contenente i file
        folder_path = "/content/audio"

        # Ottieni una lista di tutti i file nella cartella
        files = os.listdir(folder_path)

        # Ordina la lista di file in base all'ultimo accesso
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    


       






    
    def load_padCalculator(self):
        # Percorso della cartella audio
        cartella_audio = "/content/audio"

        # Ottenere la lista dei file nella cartella audio
        files_in_cartella = os.listdir(cartella_audio)

        # Se la cartella è vuota, mostra un messaggio di avviso
        if not files_in_cartella:
            print("Nessun file trovato nella cartella audio.")
        else:
            # Prendi il primo file disponibile (puoi implementare la logica per selezionare un file specifico)
            nome_file = files_in_cartella[0]

            # Percorso completo del file
            input_audio_path = os.path.join(cartella_audio, nome_file)


        # Carica l'audio dal file
        audio_data, sample_rate = librosa.load(input_audio_path, sr=None)

        # Calcola la lunghezza in millisecondi dell'audio originale
        audio_length_ms = (len(audio_data) / sample_rate) * 1000

        valore_pad = int(audio_length_ms / self.num_segments)

        return audio_length_ms,valore_pad, sample_rate,audio_data, input_audio_path

   
   
    def pad_audio(self,audio_data, target_length):
        current_length = len(audio_data)
        if current_length >= target_length:
            return audio_data
        else:
            deficit = target_length - current_length
            padding = np.zeros(deficit)
            padded_audio = np.concatenate((audio_data, padding))
            return padded_audio




    def split_audio(self,input_audio_path, output_folder, num_segments):
        # Carica il file audio
        audio = AudioSegment.from_file(input_audio_path)

    # Controlla se la cartella di output esiste, altrimenti creala
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # Calcola la durata totale dell'audio
        audio_duration = len(audio)

    # Calcola la durata approssimativa di ciascun frammento
        segment_duration = audio_duration // num_segments

    # Dividi l'audio in frammenti
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            segment = audio[start_time:end_time]

        # Costruisci il nome del file utilizzando zfill per garantire un ordinamento corretto
            output_filename = f"segment_{str(i).zfill(len(str(num_segments)))}.wav"
            output_path = os.path.join(output_folder, output_filename)

        # Salva il frammento di audio
            segment.export(output_path, format="wav")


    

def stuff_for_test(input_audio_path,num_segments,fps):
# Specifica il percorso del file audio
    audio_file_path = input_audio_path

# Carica il file audio utilizzando pydub
    audio = AudioSegment.from_file(audio_file_path)

# Ottieni la lunghezza dell'audio in secondi
    audio_length= len(audio)//1000

    audio_offsets = [0]
    timelaps = (audio_length// num_segments)
    temp = 0

    for x in range (num_segments):
        audio_offsets.append(temp+timelaps)
        temp += timelaps

# Specifica il percorso della cartella da cancellare
    folderr_path = "/content/dreams"

# Verifica se la cartella esiste prima di cancellarla
    if os.path.exists(folderr_path):
    # Cancella la cartella e tutti i suoi contenuti
        shutil.rmtree(folderr_path)
        del folderr_path

# Convert seconds to frames
    num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]  
    num_interpolation_steps = num_interpolation_steps[:-1]

    return num_interpolation_steps, audio_offsets




####################################################################################


def extract_audio_features(audio_path, num_mfcc=13):
    # Carica il file audio
    y, sr = librosa.load(audio_path, sr=None)

    # Estrai le caratteristiche MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)

    # Estrai le caratteristiche Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Estrai le caratteristiche Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Estrai le caratteristiche Chroma Energy Normalized
    chroma_energy_normalized = librosa.feature.chroma_cens(y=y, sr=sr)

    # Estrai le caratteristiche Tonnetz (Tonal Centroids)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Estrai le caratteristiche HPR (Harmonic-to-Percussive Ratio)
    harmonic, percussive = librosa.effects.hpss(y)
    hpr = np.mean(harmonic) / (np.mean(percussive) + 1e-5)  # Aggiungiamo 1e-5 per evitare divisioni per zero

    # Estrai il Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Estrai le caratteristiche del Timbro Armonico (Armonico)
    armonico = librosa.effects.harmonic(y=y)

    # Estrai le Sparsity Features
    sparsity = np.mean(np.abs(y)) / np.sqrt(np.mean(y**2))

    # Calcola la media delle feature per ciascuna categoria
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    chroma_energy_normalized_mean = np.mean(chroma_energy_normalized, axis=1)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    # Converti le caratteristiche unidimensionali in tensori bidimensionali
    hpr_tensor = torch.tensor(hpr).unsqueeze(0)
    tempo_tensor = torch.tensor(tempo).unsqueeze(0)
    armonico_mean_tensor = torch.tensor(armonico.mean()).unsqueeze(0)
    sparsity_mean_tensor = torch.tensor(sparsity).unsqueeze(0)

    # Concatena tutte le caratteristiche
    combined_features = torch.cat((torch.tensor(mfcc_mean), torch.tensor(chroma_mean), torch.tensor(spectral_contrast_mean), torch.tensor(chroma_energy_normalized_mean), torch.tensor(tonnetz_mean), hpr_tensor, tempo_tensor, armonico_mean_tensor, sparsity_mean_tensor))

    return combined_features

def extract_audio_features_from_directory(directory, num_mfcc=13, num_samples=10):
    # Inizializza una lista per contenere tutte le caratteristiche estratte
    all_features = []

    # Ottieni il numero totale di file nella directory
    num_files = len(os.listdir(directory)[:num_samples])

    # Utilizza tqdm per creare una barra di avanzamento
    with tqdm(total=num_files, desc='Extracting features') as pbar:
        # Itera su tutti i file nella directory
        for filename in os.listdir(directory)[:num_samples]:
            # Ottieni il percorso completo del file
            filepath = os.path.join(directory, filename)

            # Estrai le caratteristiche audio dal file corrente
            features = extract_audio_features(filepath, num_mfcc)

            # Aggiungi le caratteristiche estratte alla lista
            all_features.append(features)

            # Aggiorna la barra di avanzamento
            pbar.update(1)

    return all_features


