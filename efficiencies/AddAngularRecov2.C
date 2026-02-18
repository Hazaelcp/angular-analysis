#include <iostream>
#include <vector>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TLorentzVector.h"
#include "Math/Vector4D.h"
#include "Math/Vector3D.h"
#include "Math/Boost.h"

// Usamos ROOT::Math para mayor estabilidad numérica
using LV = ROOT::Math::PxPyPzMVector;
using XYZVector = ROOT::Math::XYZVector;

struct AngularVars {
    double costhetal;
    double costhetak;
    double phi;
};
// --- Función de Cálculo Angular ---
AngularVars ComputeAngles(const TLorentzVector& p4_lep_minus, const TLorentzVector& p4_lep_plus, 
                          const TLorentzVector& p4_had_K,     const TLorentzVector& p4_had_pi) 
{
    // CAMBIO DE NOMBRES PARA EVITAR CONFLICTO CON <cmath>
    const double MASS_MU = 0.105658;
    const double MASS_K  = 0.493677;
    const double MASS_PI = 0.139570; // Antes M_PI, que chocaba con pi matemático

    LV p_mu_minus(p4_lep_minus.Px(), p4_lep_minus.Py(), p4_lep_minus.Pz(), MASS_MU);
    LV p_mu_plus(p4_lep_plus.Px(),  p4_lep_plus.Py(),  p4_lep_plus.Pz(),  MASS_MU);
    LV p_K(p4_had_K.Px(),     p4_had_K.Py(),     p4_had_K.Pz(), MASS_K);
    LV p_pi(p4_had_pi.Px(),   p4_had_pi.Py(),    p4_had_pi.Pz(), MASS_PI);

    // ... (El resto de la función sigue exactamente igual) ...

    LV p_dimuon = p_mu_plus + p_mu_minus;
    LV p_Kstar  = p_K + p_pi;
    LV p_B      = p_dimuon + p_Kstar;

    // 1. CosThetaL
    ROOT::Math::Boost boost_dimuon(p_dimuon.BoostToCM());
    LV p_mu_plus_LL = boost_dimuon(p_mu_plus); 
    LV p_B_LL       = boost_dimuon(p_B);
    XYZVector v_mu_plus_LL = p_mu_plus_LL.Vect();
    XYZVector v_B_LL       = p_B_LL.Vect();
    double cos_theta_l = v_mu_plus_LL.Dot(-v_B_LL) / (v_mu_plus_LL.R() * v_B_LL.R());

    // 2. CosThetaK
    ROOT::Math::Boost boost_Kstar(p_Kstar.BoostToCM());
    LV p_K_KK = boost_Kstar(p_K);
    LV p_B_KK = boost_Kstar(p_B);
    XYZVector v_K_KK = p_K_KK.Vect();
    XYZVector v_B_KK = p_B_KK.Vect();
    double cos_theta_k = v_K_KK.Dot(-v_B_KK) / (v_K_KK.R() * v_B_KK.R());

    // 3. Phi
    ROOT::Math::Boost boost_B(p_B.BoostToCM());
    LV p_mu_plus_B  = boost_B(p_mu_plus);
    LV p_mu_minus_B = boost_B(p_mu_minus);
    LV p_K_B        = boost_B(p_K);
    LV p_pi_B       = boost_B(p_pi);
    
    XYZVector v_mu_plus_B  = p_mu_plus_B.Vect();
    XYZVector v_mu_minus_B = p_mu_minus_B.Vect();
    XYZVector v_K_B        = p_K_B.Vect();
    XYZVector v_pi_B       = p_pi_B.Vect();

    XYZVector n_L = v_mu_plus_B.Cross(v_mu_minus_B);
    XYZVector n_K = v_K_B.Cross(v_pi_B);

    double cos_phi = n_L.Dot(n_K) / (n_L.R() * n_K.R());
    if(cos_phi > 1.0) cos_phi = 1.0;
    if(cos_phi < -1.0) cos_phi = -1.0;
    double phi = std::acos(cos_phi);

    XYZVector v_Kstar_B = v_K_B + v_pi_B;
    if (v_Kstar_B.Dot(n_L.Cross(n_K)) < 0) phi = -phi;

    return {cos_theta_l, cos_theta_k, phi};
}

void AddAngularRecov2() {
    // Rutas
    std::string input_pattern = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/eficiencias/Analisys2025BdtomumuKstar_bacht2_RecoGen_myv2/BdtoKstar2Mu_KstartoKPi_MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/BdtomumuKstar_bacht2_RecoGen_myv2/251015_221737/0000/*.root/rootuple/ntuple";
    std::string output_path   = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/angular/efficiencies/datasets/RecoGenV2_Angular_Merged.root";
    
    std::cout << ">>> Initializing Analysis with Safety Cuts..." << std::endl;

    TChain* chain = new TChain("rootuple/ntuple");
    chain->Add(input_pattern.c_str());

    if (chain->GetEntries() == 0) {
        std::cerr << "!!! Error: No entries found." << std::endl;
        return;
    }

    // Punteros
    TLorentzVector* kaon_p4 = nullptr;
    TLorentzVector* pion_p4 = nullptr;
    TLorentzVector* mu1_p4 = nullptr; 
    TLorentzVector* mu2_p4 = nullptr;
    TLorentzVector* kstar_p4 = nullptr; // Necesario para el corte
    
    chain->SetBranchAddress("gen_kaon_p4", &kaon_p4);
    chain->SetBranchAddress("gen_pion_p4", &pion_p4);
    chain->SetBranchAddress("gen_muon1_p4", &mu1_p4); 
    chain->SetBranchAddress("gen_muon2_p4", &mu2_p4);
    chain->SetBranchAddress("gen_kstar_p4", &kstar_p4); // Conectamos K*

    // Preparamos salida
    TFile* outfile = new TFile(output_path.c_str(), "RECREATE");
    TTree* newtree = chain->CloneTree(0); 

    // Variables nuevas
    double cos_theta_l, cos_theta_k, phi_angle, q2_gen, mass_j_gen;

    newtree->Branch("CosThetaL_best", &cos_theta_l, "gen_cosThetaL/D");
    newtree->Branch("CosThetaK_best", &cos_theta_k, "gen_cosThetaK/D");
    newtree->Branch("Phi_best",       &phi_angle,   "gen_phi/D");
    newtree->Branch("q2",             &q2_gen,      "q2Gen/D"); 
    newtree->Branch("massJ",          &mass_j_gen,  "massJGen/D");

    Long64_t nentries = chain->GetEntries();
    Long64_t n_passed = 0; // Contador de eventos que pasan los cortes
    int report_every = 50000;

    std::cout << ">>> Processing " << nentries << " events..." << std::endl;

    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);

        if (i % report_every == 0) std::cout << "   ... event " << i << std::endl;

        // 1. Protección de punteros nulos
        if (!mu1_p4 || !mu2_p4 || !kstar_p4 || !kaon_p4 || !pion_p4) continue;

        // 2. APLICACIÓN DE FILTROS DE SEGURIDAD (Acceptance Cuts)
        //    Mu1: pT > 3.5, |eta| <= 2.5
        //    Mu2: pT > 3.5, |eta| <= 2.5
        //    K*:  pT > 0.5, |eta| <= 2.5
        
        bool pass_mu1 = (mu1_p4->Pt() > 3.5) && (std::abs(mu1_p4->Eta()) <= 2.5);
        bool pass_mu2 = (mu2_p4->Pt() > 3.5) && (std::abs(mu2_p4->Eta()) <= 2.5);
        bool pass_kstar = (kstar_p4->Pt() > 0.5) && (std::abs(kstar_p4->Eta()) <= 2.5);

        if (!pass_mu1 || !pass_mu2 || !pass_kstar) {
            continue; // Si falla algún corte, saltamos al siguiente evento (no se guarda en newtree)
        }

        // 3. Cálculos
        AngularVars angles = ComputeAngles(*mu1_p4, *mu2_p4, *kaon_p4, *pion_p4);

        cos_theta_l = angles.costhetal;
        cos_theta_k = angles.costhetak;
        phi_angle   = angles.phi;

        TLorentzVector dimuon = (*mu1_p4) + (*mu2_p4);
        q2_gen = dimuon.M2();
        mass_j_gen = dimuon.M();

        // 4. Llenado del árbol (Solo si pasó los filtros)
        newtree->Fill();
        n_passed++;
    }

    newtree->Write();
    outfile->Close();
    
    // Reporte final
    std::cout << ">>> Analysis Finished." << std::endl;
    std::cout << ">>> Total Events: " << nentries << std::endl;
    std::cout << ">>> Events Passed: " << n_passed << " (" << (double)n_passed/nentries*100.0 << "%)" << std::endl;
    std::cout << ">>> File saved: " << output_path << std::endl;
}