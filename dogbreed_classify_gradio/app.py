import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class DogBreedClassifier:
    def __init__(self, model_path="traced_models/model_tracing.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the traced model
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the same transforms used during training/testing
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Class labels
        self.labels = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']

        # Add dog breed facts dictionary
        self.breed_facts = {
            'Beagle': "Beagles have approximately 220 million scent receptors, compared to a human's mere 5 million!",
            'Boxer': "Boxers were among the first dogs to be employed as police dogs and were used as messenger dogs during wartime.",
            'Bulldog': "Despite their tough appearance, Bulldogs were bred to be companion dogs and are known for being gentle and patient.",
            'Dachshund': "Dachshunds were originally bred to hunt badgers - their name literally means 'badger dog' in German!",
            'German_Shepherd': "German Shepherds can learn a new command in as little as 5 repetitions and obey it 95% of the time.",
            'Golden_Retriever': "Golden Retrievers were originally bred as hunting dogs to retrieve waterfowl without damaging them.",
            'Labrador_Retriever': "Labs have a special water-resistant coat and a unique otter-like tail that helps them swim efficiently.",
            'Poodle': "Despite their elegant appearance, Poodles were originally water retrievers, and their fancy haircut had a practical purpose!",
            'Rottweiler': "Rottweilers are descendants of Roman drover dogs and were used to herd livestock and pull carts for butchers.",
            'Yorkshire_Terrier': "Yorkies were originally bred to catch rats in clothing mills. Despite their small size, they're true working dogs!"
        }

    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None, None
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the breed with highest probability
        max_prob_idx = torch.argmax(probabilities).item()
        predicted_breed = self.labels[max_prob_idx]
        breed_fact = self.breed_facts[predicted_breed]
        
        # Create prediction dictionary
        predictions = {
            self.labels[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }
        
        return predictions, breed_fact

classifier = DogBreedClassifier()

demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(type="pil", label="Upload a dog image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Breed Predictions"),
        gr.Textbox(label="Fun Fact About This Breed!")
    ],
    title="🐕 Dog Breed Classifier",
    description="""
    ## Identify Your Dog's Breed!
    Upload a clear photo of a dog, and I'll tell you its breed and share an interesting fact about it! 
    This model can identify 10 popular dog breeds with high accuracy.
    
    ### Supported Breeds:
    Beagle, Boxer, Bulldog, Dachshund, German Shepherd, Golden Retriever, Labrador Retriever, Poodle, Rottweiler, Yorkshire Terrier
    """,
    article="""
    ### Tips for best results:
    - Use clear, well-lit photos
    - Ensure the dog's face is visible
    - Avoid blurry or dark images
    
    Created with PyTorch and Gradio | [GitHub](your_github_link)
    """,
    examples=[
        ["examples/Beagle_56.jpg"],
        ["examples/Boxer_30.jpg"],
        ["examples/Bulldog_73.jpg"],
        ["examples/Dachshund_43.jpg"],
        ["examples/German Shepherd_57.jpg"],
        ["examples/Golden Retriever_78.jpg"],
        ["examples/Labrador Retriever_25.jpg"],
        ["examples/Poodle_85.jpg"],
        ["examples/Rottweiler_30.jpg"],
        ["examples/Yorkshire Terrier_92.jpg"]
    ],
    theme=gr.themes.Citrus(), 
    css="footer {display: none !important;}"  
)

if __name__ == "__main__":
    demo.launch() 