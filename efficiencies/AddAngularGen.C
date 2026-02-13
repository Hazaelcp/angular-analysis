#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "Math/Vector4D.h"
#include "Math/Vector3D.h"
#include "Math/Boost.h"

// Estructura para devolver los 3 ángulos juntos
struct AngularVars {
    double costhetal;
    double costhetak;
    double phi;
};

// Función matemática de cálculo de ángulos
AngularVars ComputeAngles(const TLorentzVector& mu_negative, const TLorentzVector& mu_positive, 
                          const TLorentzVector& k,   const TLorentzVector& pi) 
{
    const double MASS_MU = 0.105658;
    const double MASS_K  = 0.493677;
    const double MASS_PI = 0.139570;

    using LV = ROOT::Math::PxPyPzMVector;
    using XYZVector = ROOT::Math::XYZVector;

    // Convertir TLorentzVector a Math::Vector (más moderno y preciso para boosts)
    LV p_mu_minus(mu_negative.Px(), mu_negative.Py(), mu_negative.Pz(), MASS_MU);
    LV p_mu_plus(mu_positive.Px(),  mu_positive.Py(),  mu_positive.Pz(),  MASS_MU);
    
    LV p_K(k.Px(),     k.Py(),     k.Pz(), MASS_K);
    LV p_pi(pi.Px(),   pi.Py(),    pi.Pz(), MASS_PI);

    LV p_dimuon = p_mu_plus + p_mu_minus;
    LV p_Kstar  = p_K + p_pi;
    LV p_B      = p_dimuon + p_Kstar;

    // --- 1. CosThetaL ---
    ROOT::Math::Boost boost_to_dimuon(p_dimuon.BoostToCM());
    LV p_mu_plus_dimuon = boost_to_dimuon(p_mu_plus);
    LV p_B_dimuon       = boost_to_dimuon(p_B);
    
    XYZVector v_mu_plus_dimuon = p_mu_plus_dimuon.Vect();
    XYZVector v_B_dimuon       = p_B_dimuon.Vect();
    
    double costheta_l = v_mu_plus_dimuon.Dot(-v_B_dimuon) / (v_mu_plus_dimuon.R() * v_B_dimuon.R());

    // --- 2. CosThetaK ---
    ROOT::Math::Boost boost_to_Kstar(p_Kstar.BoostToCM());
    LV p_K_Kstar = boost_to_Kstar(p_K);
    LV p_B_Kstar = boost_to_Kstar(p_B);

    XYZVector v_K_Kstar = p_K_Kstar.Vect();
    XYZVector v_B_Kstar = p_B_Kstar.Vect();

    double costheta_k = v_K_Kstar.Dot(-v_B_Kstar) / (v_K_Kstar.R() * v_B_Kstar.R());

    // --- 3. Phi ---
    ROOT::Math::Boost boost_to_B(p_B.BoostToCM());
    LV p_mu_plus_B  = boost_to_B(p_mu_plus);
    LV p_mu_minus_B = boost_to_B(p_mu_minus);
    LV p_K_B        = boost_to_B(p_K);
    LV p_pi_B       = boost_to_B(p_pi);
    LV p_Kstar_B    = boost_to_B(p_Kstar); 

    XYZVector v_mu_plus_B  = p_mu_plus_B.Vect();
    XYZVector v_mu_minus_B = p_mu_minus_B.Vect();
    XYZVector v_K_B        = p_K_B.Vect();
    XYZVector v_pi_B       = p_pi_B.Vect();
    XYZVector v_Kstar_B    = p_Kstar_B.Vect();

    XYZVector n_L = v_mu_plus_B.Cross(v_mu_minus_B);
    XYZVector n_K = v_K_B.Cross(v_pi_B);

    double phi = std::acos(n_L.Dot(n_K) / (n_L.R() * n_K.R()));

    // Signo de Phi
    double sign_check = v_Kstar_B.Dot(n_L.Cross(n_K));
    if (sign_check < 0) phi = -phi;

    return {costheta_l, costheta_k, phi};
}

void AddAngularGen() {
    // --- RUTAS DE ENTRADA Y SALIDA ---
    std::string input_pattern = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/eficiencias/Analisys2025BdtomumuKstar_bacht2_OnlyGen_myv5/BdtoKstar2Mu_KstartoKPi_TuneCP5_13p6TeV_pythia8-evtgen/BdtomumuKstar_bacht2_OnlyGen_myv5/251005_190941/0000/*.root";
    std::string output_path   = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/angular/efficiencies/datasets/GenLevel_Angular_Merged.root";
    std::string tree_name     = "rootuple/ntuple";

    std::cout << ">>> Initializing Analysis..." << std::endl;

    // Cargar cadena de archivos
    TChain* chain = new TChain(tree_name.c_str());
    chain->Add(input_pattern.c_str());

    if (chain->GetEntries() == 0) {
        std::cerr << "!!! Error: No entries found in path: " << input_pattern << std::endl;
        return;
    }

    // Punteros para leer las ramas del TLorentzVector
    TLorentzVector* b_p4 = nullptr;
    TLorentzVector* kstar_p4 = nullptr;
    TLorentzVector* kaon_p4 = nullptr;
    TLorentzVector* pion_p4 = nullptr;
    TLorentzVector* mu1_p4 = nullptr;
    TLorentzVector* mu2_p4 = nullptr;
    
    // Conectar ramas
    chain->SetBranchAddress("gen_b_p4", &b_p4);
    chain->SetBranchAddress("gen_kstar_p4", &kstar_p4);
    chain->SetBranchAddress("gen_kaon_p4", &kaon_p4);
    chain->SetBranchAddress("gen_pion_p4", &pion_p4);
    chain->SetBranchAddress("gen_muon1_p4", &mu1_p4);
    chain->SetBranchAddress("gen_muon2_p4", &mu2_p4);

    // Preparar archivo de salida
    // Clonamos la estructura del árbol original (CloneTree(0)) para mantener las variables viejas
    // y agregamos las nuestras.
    TFile* outfile = new TFile(output_path.c_str(), "RECREATE");
    TTree* newtree = chain->CloneTree(0);

    // --- VARIABLES NUEVAS ---
    double cos_theta_l;
    double cos_theta_k;
    double phi_angle;
    double q2_gen;       // Momento transferido al cuadrado
    double mass_j_gen;   // Masa invariante dimuón (massJ)

    // Crear las ramas nuevas en el árbol de salida
    newtree->Branch("gen_cosThetaL", &cos_theta_l, "gen_cosThetaL/D");
    newtree->Branch("gen_cosThetaK", &cos_theta_k, "gen_cosThetaK/D");
    newtree->Branch("gen_phi",       &phi_angle,   "gen_phi/D");
    newtree->Branch("q2Gen",         &q2_gen,      "q2Gen/D"); 
    newtree->Branch("massJGen",      &mass_j_gen,  "massJGen/D"); // <--- Nueva Rama MassJ

    Long64_t nentries = chain->GetEntries();
    int report_every = 100000;

    std::cout << ">>> Processing " << nentries << " events..." << std::endl;

    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);

        if (i % report_every == 0) {
            std::cout << "   ... event " << i << " / " << nentries 
                      << " (" << (int)((double)i/nentries*100) << "%)" << std::endl;
        }

        // Protección contra punteros nulos
        if (!mu1_p4 || !mu2_p4 || !kaon_p4 || !pion_p4) continue;

        // --- CÁLCULOS FÍSICOS ---
        
        // 1. Reconstruir dimuón
        TLorentzVector dimuon = (*mu1_p4) + (*mu2_p4);
        
        // 2. Calcular cinemática
        q2_gen = dimuon.M2();     // Masa al cuadrado
        mass_j_gen = dimuon.M();  // Masa invariante (sqrt(q2))

        // 3. Calcular Ángulos
        // Asumimos mu1 negativo y mu2 positivo (falta de ID en OnlyGen)
        AngularVars angles = ComputeAngles(*mu1_p4, *mu2_p4, *kaon_p4, *pion_p4);

        cos_theta_l = angles.costhetal;
        cos_theta_k = angles.costhetak;
        phi_angle   = angles.phi;

        newtree->Fill();
    }

    newtree->Write();
    outfile->Close();
    std::cout << ">>> Analysis Finished Successfully." << std::endl;
    std::cout << ">>> File saved: " << output_path << std::endl;
}