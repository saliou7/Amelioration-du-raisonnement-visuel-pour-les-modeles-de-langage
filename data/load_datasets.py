import os
import requests
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import datasets
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

class LaionDatasetLoader:
    def __init__(self, args, tokenizer: AutoTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
        # Configuration des chemins
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / "data_laion/images"
        self.metadata_dir = self.base_dir / "data_laion/metadata"
        self.dataset_path = self.base_dir / "datasets/processed_dataset.hf"
        
        # Création des répertoires
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_image_path(self, url: str) -> Path:
        """Génère un chemin de fichier unique à partir de l'URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.image_dir / f"{url_hash}.jpg"

    def _download_image(self, url: str, max_retries: int = 3) -> Optional[Path]:
        """Télécharge une image avec reprise sur erreur"""
        save_path = self._get_image_path(url)
        
        if save_path.exists():
            return save_path
            
        for _ in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return save_path
            except Exception as e:
                print(f"Échec du téléchargement (tentative {_+1}/{max_retries}): {url} - {str(e)}")
        
        print(f"Abandon du téléchargement après {max_retries} tentatives: {url}")
        return None

    def download_images(self, urls: List[str]):
        """Télécharge toutes les images en amont"""
        progress = tqdm(urls, desc="Téléchargement des images")
        for url in progress:
            self._download_image(url)
            progress.set_postfix({"status": f"{len(os.listdir(self.image_dir))} images"})

    def _tokenize_example(self, example: Dict) -> Optional[Dict]:
        """Tokenise un exemple après vérification de l'image"""
        image_path = self._get_image_path(example['url'])
        
        if not image_path.exists():
            return None
            
        text_inputs = self.tokenizer(
            example['caption'] if not self.args.short else example['short_caption'],
            max_length=self.args.max_seq_length,
            padding="max_length",
            truncation=True
        )
        
        return {
            **text_inputs,
            "image_path": str(image_path),
            "url_hash": hashlib.md5(example['url'].encode()).hexdigest()
        }

    def process_dataset(self, raw_dataset: Dataset) -> Dataset:
        """Processus complet de traitement des données"""
        # Étape 1: Téléchargement de toutes les images
        self.download_images(raw_dataset['url'])
        
        # Étape 2: Filtrage et tokenisation
        processed_data = []
        for example in tqdm(raw_dataset, desc="Traitement des données"):
            processed = self._tokenize_example(example)
            if processed:
                processed_data.append(processed)
        
        return Dataset.from_list(processed_data)

    def process_existing_images(self, raw_dataset: Dataset) -> Dataset:
        """Traite uniquement les images déjà téléchargées par plus petits lots"""
        existing_images = {f.stem: f for f in self.image_dir.glob("*.jpg")}
        print(f"Nombre d'images existantes: {len(existing_images)}")
        
        # Ajout de compteurs pour debug
        image_usage_count = {}
        
        # Réduire la taille des lots
        batch_size = 250
        chunk_size = 10000  # Taille des grands morceaux
        processed_datasets = []
        
        # Traitement par grands morceaux
        for chunk_start in range(0, len(raw_dataset), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(raw_dataset))
            chunk = raw_dataset.select(range(chunk_start, chunk_end))
            chunk_processed_data = []
            
            # Traitement par petits lots dans chaque morceau
            for i in range(0, len(chunk), batch_size):
                batch = chunk.select(range(i, min(i + batch_size, len(chunk))))
                processed_batch = []
                
                for example in tqdm(batch, 
                    desc=f"Lot {i//batch_size + 1} du morceau {chunk_start//chunk_size + 1}"):
                    url_hash = hashlib.md5(example['url'].encode()).hexdigest()
                    if url_hash in existing_images:
                        # Compte combien de fois chaque image est utilisée
                        image_usage_count[url_hash] = image_usage_count.get(url_hash, 0) + 1
                        
                        text_inputs = self.tokenizer(
                            example['caption'] if not self.args.short else example['short_caption'],
                            max_length=self.args.max_seq_length,
                            padding="max_length",
                            truncation=True
                        )
                        processed_batch.append({
                            **text_inputs,
                            "image_path": str(existing_images[url_hash]),
                            "url_hash": url_hash
                        })
                
                if processed_batch:
                    chunk_processed_data.extend(processed_batch)
                
                del batch
                del processed_batch
            
            # Sauvegarder le morceau traité
            if chunk_processed_data:
                chunk_dataset = Dataset.from_list(chunk_processed_data)
                chunk_path = self.dataset_path.parent / f"chunk_{chunk_start//chunk_size}.hf"
                chunk_dataset.save_to_disk(str(chunk_path))
                processed_datasets.append(chunk_path)
                print(f"Morceau {chunk_start//chunk_size + 1} sauvegardé: {len(chunk_processed_data)} exemples")
            
            del chunk
            del chunk_processed_data
        
        # Combiner tous les morceaux
        print("Combinaison des morceaux...")
        final_dataset = datasets.concatenate_datasets([
            datasets.load_from_disk(str(path)) 
            for path in processed_datasets
        ])
        
        # Affiche les statistiques
        print("\nStatistiques d'utilisation des images:")
        print(f"Images utilisées une fois: {sum(1 for count in image_usage_count.values() if count == 1)}")
        print(f"Images utilisées deux fois: {sum(1 for count in image_usage_count.values() if count == 2)}")
        print(f"Images utilisées plus de deux fois: {sum(1 for count in image_usage_count.values() if count > 2)}")
        
        # Nettoyage
        for path in processed_datasets:
            if path.exists():
                import shutil
                shutil.rmtree(path)
        
        print(f"Nombre total d'exemples traités: {len(final_dataset)}")
        return final_dataset

    def load(self) -> Dataset:
        """Charge ou génère le dataset traité"""
        if self.dataset_path.exists() and not self.args.overwrite_cache:
            return datasets.load_from_disk(str(self.dataset_path))
            
        # Téléchargement du dataset brut
        raw_dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split='train')
        
        if self.args.max_train_samples:
            raw_dataset = raw_dataset.select(range(self.args.max_train_samples))
        
        # Utilise uniquement les images existantes
        processed_dataset = self.process_existing_images(raw_dataset)
        processed_dataset.save_to_disk(str(self.dataset_path))
        
        return processed_dataset

def load_laion_220(args, tokenizer: AutoTokenizer) -> Dataset:
    """Fonction d'interface principale"""
    loader = LaionDatasetLoader(args, tokenizer)
    return loader.load()