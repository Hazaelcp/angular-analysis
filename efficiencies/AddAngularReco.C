#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "TMath.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TMatrixD.h"
#include "TChain.h"
#include "TFile.h"
#include <ROOT/RDataFrame.hxx>
#include "ROOT/RDFHelpers.hxx"
#include "Math/Vector4D.h"
#include "Math/Vector3D.h"
#include "Math/GenVector/Boost.h"

using namespace std;
using namespace ROOT;

// =================================================================================
// 1. DECLARACIONES ADELANTADAS (PROTOTIPOS)
// =================================================================================

std::vector<double> ComputeAngles(
    double mu1_px, double mu1_py, double mu1_pz, int mu1_ch,
    double mu2_px, double mu2_py, double mu2_pz, int mu2_ch,
    double trk1_px, double trk1_py, double trk1_pz, int trk1_ch,
    double trk2_px, double trk2_py, double trk2_pz, int trk2_ch,
    int hypothesis 
);

ROOT::RDF::RNode AddSelection(ROOT::RDF::RNode node, std::string f1, std::string f2, std::string f3, std::string f4);
ROOT::RDF::RNode SetVariables(ROOT::RDF::RNode node, UInt_t WhatSample);

TVector3 myTV3(const double &x, const double &y, const double &z);
TVector3 myTV3_2(const double &x, const double &y, const double &z);
std::vector<std::vector<double>> myTM(const double &x, const double &y, const double &z, const double &xy, const double &xz, const double &yz);
std::vector<double> MassVeto(const double &x1, const double &y1, const double &z1, const double &x2, const double &y2, const double &z2);
Double_t deltaR_(const TVector3 &v1, const TVector3 &v2);

std::vector<double> Lifetime(TVector3 &pv, TVector3 &sv, vector<vector<double>> &EPV_i, vector<vector<double>> &ESV_i, TVector3 &pT, Double_t &M);

std::vector<double> MydeltaR( const TVector3 &Mu1, const TVector3 &Mu2, const TLorentzVector genmu1, const TLorentzVector genmu2, 
                              const int &mu_charge1, const int &mu_charge2, const TVector3 &Pi1, const TVector3 &Pi2, 
                              const TLorentzVector genk, const TLorentzVector genpi, const int &pi_charge1, const int &pi_charge2);


// =================================================================================
// 2. FUNCIÓN PRINCIPAL (MAIN)
// =================================================================================

void AddAngularReco( UInt_t year=2023, UInt_t sample=2, UInt_t era=1)
{  
    ROOT::EnableImplicitMT(8); 
    
    std::cout << ">>> [AddAngularReco] Iniciando análisis..." << std::endl;

    TChain tree("ntuple");

    // Configuración de rutas
    if(year==2023 && sample==1){
        if(era==1) tree.Add("Rootuple_BdtoJpsiKstar_Data2022.root/rootuple/ntuple");
        else tree.Add("Data2023/.../Rootuple_BdtoJpsiKstar_Data2022_1.root/rootuple/ntuple");
    }
    else if(year==2023 && sample==2) {
        // Placeholder
    }
    else if (year==2022 && sample==1){
        tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/Analisys2025/BdJpsiKstar_miniAOD_2022Gv1/ParkingDoubleMuonLowMass1/BdJpsiKstar_2022Gv1_ParkingDoubleMuonLowMass1-Run2022G-PromptReco-v1/250627_043446/0000/*.root/rootuple/ntuple");
        cout << " Running for DATA \n";
    }
    else if(year==2022 && sample==2){
        // RUTA MC RECO-GEN
        tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/eficiencias/Analisys2025BdtomumuKstar_bacht2_RecoGen_myv1/BdtoKstar2Mu_KstartoKPi_MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/BdtomumuKstar_bacht2_RecoGen_myv1/251003_201821/0000/*.root/rootuple/ntuple");
        cout << " Running for RecoGen Sample (MC 2022) \n";
    }
    else if (year==2022 && sample==3){
        tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/Analisys2025BdtoKstarPsi2s_bacht2_myv1/.../*.root/rootuple/ntuple");
        cout << " Running for Psi(2S) \n";
    }
    else if(year==2022 && sample==4){
        tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/Analisys2025BdtomumuKstar_bacht2_myv2/.../*.root/rootuple/ntuple");
        cout << " Running for NoRes\n";
    }
    else {
        std::cout << "!!! Error: No valid file configuration found." << std::endl;
        return;
    }

    ROOT::RDataFrame dDF(tree);
    auto nentries = dDF.Count();
    std::cout << ">>> Total entries found: " << *nentries << std::endl;

    // Trigger Filter (Técnico, no físico)
    auto dDF1 = dDF.Filter("tri_DMu4_LM_Displaced==1 || tri_DMu4_3_LM==1", "selTrigger");
    
    // Constantes
    Double_t MpdgB = 5.2796;
    Double_t Mpdgv0 = 0.4976;
    Double_t Mpdgmu = 0.105658;
    // Masa PDG del K* para la selección
    const double MpdgKstar = 0.89166; 

    auto dDF3 = dDF1
        .Define("B",    myTV3, {"B1_px", "B1_py", "B1_pz"})
        .Define("Jpsi", myTV3, {"B_J_px", "B_J_py", "B_J_pz"})
        .Define("mu1",  myTV3, {"B_J_px1", "B_J_py1", "B_J_pz1"})
        .Define("mu2",  myTV3, {"B_J_px2", "B_J_py2", "B_J_pz2"})
        .Define("pi1",  myTV3, {"B_Trk_px1", "B_Trk_py1", "B_Trk_pz1"})
        .Define("pi2",  myTV3, {"B_Trk_px2", "B_Trk_py2", "B_Trk_pz2"})
        .Define("pT",   myTV3_2, {"B1_px", "B1_py", "B1_pz"})
        .Define("pv",   myTV3, {"priVtxX", "priVtxY", "priVtxZ"})
        .Define("sv",   myTV3, {"B_DecayVtxX", "B_DecayVtxY", "B_DecayVtxZ"})
        .Define("EPV",  myTM, {"priVtxXE", "priVtxYE", "priVtxZE", "priVtxXYE", "priVtxXZE", "priVtxYZE"})
        .Define("ESV",  myTM, {"B_DecayVtxXE", "B_DecayVtxYE", "B_DecayVtxZE", "B_DecayVtxXYE", "B_DecayVtxXZE", "B_DecayVtxYZE"})
        .Define("massveto", MassVeto, {"B_Trk_px1", "B_Trk_py1", "B_Trk_pz1","B_Trk_px2", "B_Trk_py2", "B_Trk_pz2"})
        .Define("MpdgB", [&MpdgB] { return MpdgB; })
        .Define("Mpdgv0", [&Mpdgv0] { return Mpdgv0; })
        .Define("Mpdgmu", [&Mpdgmu] { return Mpdgmu; })
        
        // --- ANGULAR RECO (Hyp 1 & 2) ---
        .Define("Hyp1_Flag", [](){ return 1; }) 
        .Define("Hyp2_Flag", [](){ return 2; })
        .Define("Angulos_Hyp1", ComputeAngles, {
            "B_J_px1", "B_J_py1", "B_J_pz1", "B_J_charge1",
            "B_J_px2", "B_J_py2", "B_J_pz2", "B_J_charge2",
            "B_Trk_px1", "B_Trk_py1", "B_Trk_pz1", "B_Trk_charge1",
            "B_Trk_px2", "B_Trk_py2", "B_Trk_pz2", "B_Trk_charge2",
            "Hyp1_Flag" 
        })
        .Define("Angulos_Hyp2", ComputeAngles, {
            "B_J_px1", "B_J_py1", "B_J_pz1", "B_J_charge1",
            "B_J_px2", "B_J_py2", "B_J_pz2", "B_J_charge2",
            "B_Trk_px1", "B_Trk_py1", "B_Trk_pz1", "B_Trk_charge1",
            "B_Trk_px2", "B_Trk_py2", "B_Trk_pz2", "B_Trk_charge2",
            "Hyp2_Flag" 
        });

    auto dDF4 = dDF3.Define("myct", Lifetime, {"pv","sv","EPV","ESV","pT","MpdgB"});

    auto dDF5 = dDF4
        .Define("dxB", "B_DecayVtxX - priVtxX")
        .Define("dyB", "B_DecayVtxY - priVtxY")
        .Define("dxBE", "B_DecayVtxXE")
        .Define("dyBE", "B_DecayVtxYE")
        .Define("sigLxyBtmp", "(dxB*dxB +dyB*dyB)/sqrt( dxB*dxB*dxBE*dxBE + dyB*dyB*dyBE*dyBE )")
        .Define("cosAlphaXYb", "( B1_px*dxB + B1_py*dyB )/( sqrt(dxB*dxB+dyB*dyB)*B.Pt()  )");
  
    // Variables dummy para selección (NO SE APLICARÁN)
    auto selDimuon = "mu1.Pt()>=4.0"; 
    auto selJmass = "B_J_mass>=0.2";
    auto selOthers = "B_J_Prob>0.01";
    auto selOthers2 = "B_TrkTrk_mass1>0.5";
  
    // Aplicar selección "Empty" (Pass-through para eficiencias)
    auto dDF6 = AddSelection(dDF5, selDimuon, selJmass, selOthers, selOthers2);

    auto seldata = SetVariables(dDF6, sample);
    
    // Columnas a guardar
    vector<string> columns  = {
            "massB1","massB2","masskstar1","masskstar2","massvetopipi","massvetokk","massJ","cosalfaB","Bdl", "BdlE","Beta","Bpt","Jpsipt"
           ,"Pi1pt","Pi2pt","mu1pt", "mu2pt", "muon_dca", "Jprob","Bprob","Jeta","mu1eta","mu2eta","Pi1eta","Pi2eta","mu1phi","mu2phi"
           ,"Pi1phi","Pi2phi", "event", "run", "lumiblock", "nVtx","priVtxCL", "tri_DMu4_LM_Displaced","tri_DMu4_3_LM","min_dr_trk1_muons"
           ,"min_dr_trk2_muons","min_dpt_trk1_muons","min_dpt_trk2_muons","mu1medium","mu2medium","Pi1charge","Pi2charge","mu1charge"
           ,"mu2charge","mu1pi1DR","mu1pi2DR","mu2pi1DR","mu2pi2DR","Frompv_Trk1","Frompv_Trk2", 
           // Hipótesis puras
           "CosThetaL_H1", "CosThetaK_H1", "Phi_H1",
           "CosThetaL_H2", "CosThetaK_H2", "Phi_H2",
           // Variables SELECCIONADAS (BEST MATCH)
           "massKstar_best", "massB_best", "CosThetaK_best", "CosThetaL_best", "Phi_best"
    };

    vector<string> columnsMC  = {
        "massBGen","BptGen","BetaGen","massJGen","JpsiptGen","mu1ptGen","mu2ptGen"
        ,"masskstarGen","kstarptGen","drmu1","drmu2","pi1ptGen","pi2ptGen","dr_track1K","dr_track2Pi"
        ,"dr_track1Pi","dr_track2K"
    };
    if(sample!=1){ columns.insert(columns.end(), columnsMC.begin(), columnsMC.end());}
  
    // =========================================================================
    // SALIDA
    // =========================================================================
    std::string output_path = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/angular/efficiencies/datasets/RecoGenLevel_Angular_Merged.root";
    
    std::cout << ">>> Saving Snapshot to: " << output_path << std::endl;
    
    // Snapshot
    seldata.Snapshot("treeBd", output_path, columns);
    
    std::cout << ">>> AddAngularReco Finished Successfully." << std::endl;
}

// =================================================================================
// 3. IMPLEMENTACIÓN DE FUNCIONES AUXILIARES
// =================================================================================

// --- SetVariables (ACTUALIZADO CON SELECCIÓN DE MEJOR HIPÓTESIS) ---
ROOT::RDF::RNode SetVariables(ROOT::RDF::RNode node, UInt_t WhatSample){
  
  // Constante para selección (Masa K* PDG)
  const double MpdgKstar = 0.89166;

  // Nodo base común para Data y MC
  auto base_node = node
      .Define("massB1","B1_mass")
      .Define("massB2","B2_mass")
      .Define("massvetopipi","massveto[0]")
      .Define("massvetokk","massveto[1]")
      .Define("masskstar1","B_TrkTrk_mass1")
      .Define("masskstar2","B_TrkTrk_mass2")
      .Define("massJ","B_J_mass")
      .Define("cosalfaB","cosAlphaXYb")
      .Define("Bdl","myct[0]")
      .Define("BdlE","myct[1]")
      .Define("Beta","B.Eta()")
      .Define("Bpt","B.Pt()")
      .Define("Pi1pt","pi1.Pt()")
      .Define("Pi2pt","pi2.Pt()")
      .Define("Jpsipt","Jpsi.Pt()")
      .Define("mu1pt","mu1.Pt()")
      .Define("mu2pt","mu2.Pt()")
      .Define("Jprob","B_J_Prob")
      .Define("Bprob","B1_Prob")
      .Define("Jeta","Jpsi.Eta()")
      .Define("mu1eta","mu1.Eta()")
      .Define("mu2eta","mu2.Eta()")
      .Define("Pi1eta","pi1.Eta()")
      .Define("Pi2eta","pi2.Eta()")
      .Define("mu1phi","mu1.Phi()")
      .Define("mu2phi","mu2.Phi()")
      .Define("Pi1phi","pi1.Phi()")
      .Define("Pi2phi","pi2.Phi()")
      .Define("Pi1charge","B_Trk_charge1")
      .Define("Pi2charge","B_Trk_charge2")
      .Define("mu1charge","B_J_charge1")
      .Define("mu2charge","B_J_charge2")
      .Define("mu1pi1DR",deltaR_, {"mu1","pi1"})
      .Define("mu1pi2DR",deltaR_, {"mu1","pi2"})
      .Define("mu2pi1DR",deltaR_, {"mu2","pi1"})
      .Define("mu2pi2DR",deltaR_, {"mu2","pi2"})
      .Define("Frompv_Trk1","FrompvTrk1")
      .Define("Frompv_Trk2","FrompvTrk2")
      // Angulos H1
      .Define("CosThetaL_H1", "Angulos_Hyp1[0]")
      .Define("CosThetaK_H1", "Angulos_Hyp1[1]")
      .Define("Phi_H1",       "Angulos_Hyp1[2]")
      // Angulos H2
      .Define("CosThetaL_H2", "Angulos_Hyp2[0]")
      .Define("CosThetaK_H2", "Angulos_Hyp2[1]")
      .Define("Phi_H2",       "Angulos_Hyp2[2]")

      // =====================================================================
      // AQUÍ AGREGAMOS LA SELECCIÓN DE LAS "MEJORES VARIABLES" (BEST)
      // =====================================================================
      
      // 1. Mejor Masa K*
      .Define("massKstar_best", [MpdgKstar](double mk1, double mk2) {
          double diff1 = std::abs(mk1 - MpdgKstar);
          double diff2 = std::abs(mk2 - MpdgKstar);
          return (diff1 <= diff2) ? mk1 : mk2;
      }, {"masskstar1", "masskstar2"})

      // 2. Mejor Masa B (Consistente con la decisión de arriba)
      .Define("massB_best", [MpdgKstar](double mk1, double mk2, double mb1, double mb2) {
          double diff1 = std::abs(mk1 - MpdgKstar);
          double diff2 = std::abs(mk2 - MpdgKstar);
          return (diff1 <= diff2) ? mb1 : mb2;
      }, {"masskstar1", "masskstar2", "massB1", "massB2"})

      // 3. Mejor CosThetaK
      .Define("CosThetaK_best", [MpdgKstar](double mk1, double mk2, double ctk1, double ctk2) {
          double diff1 = std::abs(mk1 - MpdgKstar);
          double diff2 = std::abs(mk2 - MpdgKstar);
          return (diff1 <= diff2) ? ctk1 : ctk2;
      }, {"masskstar1", "masskstar2", "CosThetaK_H1", "CosThetaK_H2"})

      // 4. Mejor CosThetaL
      .Define("CosThetaL_best", [MpdgKstar](double mk1, double mk2, double ctl1, double ctl2) {
          double diff1 = std::abs(mk1 - MpdgKstar);
          double diff2 = std::abs(mk2 - MpdgKstar);
          return (diff1 <= diff2) ? ctl1 : ctl2;
      }, {"masskstar1", "masskstar2", "CosThetaL_H1", "CosThetaL_H2"})

      // 5. Mejor Phi
      .Define("Phi_best", [MpdgKstar](double mk1, double mk2, double phi1, double phi2) {
          double diff1 = std::abs(mk1 - MpdgKstar);
          double diff2 = std::abs(mk2 - MpdgKstar);
          return (diff1 <= diff2) ? phi1 : phi2;
      }, {"masskstar1", "masskstar2", "Phi_H1", "Phi_H2"});


  // Ramas extra si es MC
  if (WhatSample != 1){
    return base_node
      .Define("massBGen","gen_b_p4.M()")
      .Define("BptGen","gen_b_p4.Pt()")
      .Define("BetaGen","gen_b_p4.Rapidity()")
      .Define("massJGen","gen_jpsi_p4.M()")
      .Define("JpsiptGen","gen_jpsi_p4.Pt()")
      .Define("mu1ptGen","gen_muon1_p4.Pt()")
      .Define("mu2ptGen","gen_muon2_p4.Pt()")
      .Define("masskstarGen","gen_kstar_p4.M()")
      .Define("kstarptGen","gen_kstar_p4.Pt()")
      .Define("pi1ptGen","gen_kaon_p4.Pt()")
      .Define("pi2ptGen","gen_pion_p4.Pt()")
      .Define("mudeltavector", MydeltaR,{"mu1", "mu2", "gen_muon1_p4","gen_muon2_p4","B_J_charge1","B_J_charge2","pi1", "pi2", "gen_kaon_p4","gen_pion_p4","B_Trk_charge1","B_Trk_charge2"})
      .Define("drmu1","mudeltavector[0]")
      .Define("drmu2","mudeltavector[1]")
      .Define("dr_track1K","mudeltavector[2]")
      .Define("dr_track2Pi","mudeltavector[3]")
      .Define("dr_track1Pi","mudeltavector[4]")
      .Define("dr_track2K","mudeltavector[5]");
  }
  return base_node;
}

// --- ComputeAngles ---
std::vector<double> ComputeAngles(
    double mu1_px, double mu1_py, double mu1_pz, int mu1_ch,
    double mu2_px, double mu2_py, double mu2_pz, int mu2_ch,
    double trk1_px, double trk1_py, double trk1_pz, int trk1_ch,
    double trk2_px, double trk2_py, double trk2_pz, int trk2_ch,
    int hypothesis) 
{
    const double m_mu = 0.105658;
    const double m_K  = 0.493677;
    const double m_pi = 0.139570;

    using LV = ROOT::Math::PxPyPzMVector;
    using XYZVector = ROOT::Math::XYZVector;

    LV p_mu1(mu1_px, mu1_py, mu1_pz, m_mu);
    LV p_mu2(mu2_px, mu2_py, mu2_pz, m_mu);

    LV p_mu_plus  = (mu1_ch > 0) ? p_mu1 : p_mu2;
    LV p_mu_minus = (mu1_ch > 0) ? p_mu2 : p_mu1;
    LV p_dimuon   = p_mu_plus + p_mu_minus;

    LV p_trk1, p_trk2, p_K, p_pi;

    if (hypothesis == 1) {
        p_trk1.SetCoordinates(trk1_px, trk1_py, trk1_pz, m_pi); 
        p_trk2.SetCoordinates(trk2_px, trk2_py, trk2_pz, m_K);  
        p_pi = p_trk1; p_K  = p_trk2;
    } else {
        p_trk1.SetCoordinates(trk1_px, trk1_py, trk1_pz, m_K);  
        p_trk2.SetCoordinates(trk2_px, trk2_py, trk2_pz, m_pi); 
        p_K  = p_trk1; p_pi = p_trk2;
    }

    LV p_Kstar = p_K + p_pi;
    LV p_B = p_dimuon + p_Kstar; 

    // CosThetaL
    ROOT::Math::Boost boost_to_dimuon(p_dimuon.BoostToCM());
    LV p_mu_plus_dimuon = boost_to_dimuon(p_mu_plus);
    LV p_B_dimuon       = boost_to_dimuon(p_B);
    XYZVector v_mu_plus_dimuon = p_mu_plus_dimuon.Vect();
    XYZVector v_B_dimuon       = p_B_dimuon.Vect();
    double costheta_l = v_mu_plus_dimuon.Dot(-v_B_dimuon) / (v_mu_plus_dimuon.R() * v_B_dimuon.R());

    // CosThetaK
    ROOT::Math::Boost boost_to_Kstar(p_Kstar.BoostToCM());
    LV p_K_Kstar = boost_to_Kstar(p_K);
    LV p_B_Kstar = boost_to_Kstar(p_B);
    XYZVector v_K_Kstar = p_K_Kstar.Vect();
    XYZVector v_B_Kstar = p_B_Kstar.Vect();
    double costheta_k = v_K_Kstar.Dot(-v_B_Kstar) / (v_K_Kstar.R() * v_B_Kstar.R());

    // Phi
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

    double sign_check = v_Kstar_B.Dot(n_L.Cross(n_K));
    if (sign_check < 0) phi = -phi;

    return {costheta_l, costheta_k, phi};
}

// --- AddSelection (PASS-THROUGH) ---
ROOT::RDF::RNode AddSelection( ROOT::RDF::RNode node, std::string filter1, std::string filter2, std::string filter3, std::string filter4){
  return node;
}

// --- Helpers Matemáticos ---
TVector3 myTV3(const double &x, const double &y, const double &z){
    return TVector3(x,y,z);  
}

TVector3 myTV3_2(const double &x, const double &y, const double &z){
    return TVector3(x,y,0);  
}

std::vector<std::vector<double>> myTM(const double &x, const double &y, const double &z, const double &xy, const double &xz, const double &yz){
  std::vector<std::vector<double>> theM{{0,0,0},{0,0,0},{0,0,0}};
  theM.at(0).at(0) = x; theM.at(1).at(1) = y; theM.at(2).at(2) = z;
  theM.at(0).at(1) = xy; theM.at(0).at(2) = xz; theM.at(1).at(2) = yz;
  return theM;
}

Double_t deltaR_(const TVector3 &v1, const TVector3 &v2){
    Double_t p1 = v1.Phi(); Double_t p2 = v2.Phi();
    Double_t e1 = v1.Eta(); Double_t e2 = v2.Eta();
    Double_t dp = abs(p1 - p2);
    if (dp > 3.141592) dp -= (2 * 3.14592);
    return sqrt((e1 - e2) * (e1 - e2) + dp * dp);
};

std::vector<double> MassVeto(const double &x1, const double &y1, const double &z1, const double &x2, const double &y2, const double &z2)
{
  TLorentzVector pi1,pi2, pipi;
  pi1.SetXYZM(x1, y1, z1, 0.13957);
  pi2.SetXYZM(x2, y2, z2, 0.13957);
  pipi = pi1+pi2;
  
  TLorentzVector k1,k2, kk;
  k1.SetXYZM(x1, y1, z1, 0.493677);
  k2.SetXYZM(x2, y2, z2, 0.493677);
  kk = k1+k2;

  std::vector<double> myMveto;
  myMveto.push_back( pipi.M() );
  myMveto.push_back( kk.M() );
  return myMveto;
}

std::vector<double> Lifetime( TVector3 &pv, TVector3 &sv, vector<vector<double>> &EPV_i, vector<vector<double>> &ESV_i, TVector3 &pT, Double_t &M)
{
  TMatrixD EPV(3,3), ESV(3,3);
  EPV(0,0)=EPV_i[0][0]; EPV(1,1)=EPV_i[1][1]; EPV(2,2)=EPV_i[2][2]; EPV(0,1)=EPV_i[0][1]; EPV(0,2)=EPV_i[0][2]; EPV(1,2)=EPV_i[1][2];
  ESV(0,0)=ESV_i[0][0]; ESV(1,1)=ESV_i[1][1]; ESV(2,2)=ESV_i[2][2]; ESV(0,1)=ESV_i[0][1]; ESV(0,2)=ESV_i[0][2]; ESV(1,2)=ESV_i[1][2];

  TVector3 svT(sv.X(),sv.Y(),0.0);
  TVector3 pvT(pv.X(),pv.Y(),0.0);
  TVector3 d = svT - pvT;

  TMatrixD VSV(2,2), VPV(2,2);
  VSV(0,0)=ESV(0,0); VSV(1,1)=ESV(1,1); VSV(1,0)=ESV(1,0); VSV(0,1)=VSV(1,0);
  VPV(0,0)=EPV(0,0); VPV(1,1)=EPV(1,1); VPV(1,0)=EPV(1,0); VPV(0,1)=VPV(1,0);

  TMatrixD VL(2,2); VL = VSV; VL+=VPV; 
  TVector3 p = pT;

  TMatrixD VP(2,2); VP(0,0)=0.0; VP(1,1)=0.0; VP(0,1)=0.0; VP(1,0)=0.0;  

  double Lxy = d.Dot(p)/p.Mag();
  double lf = Lxy*M/p.Mag();
  double ct =  lf;

  // Errors
  TMatrixD A(2,2), B(2,2), C(2,2);
  A(0,0) = p.X()*p.X()/p.Mag2(); A(1,1) = p.Y()*p.Y()/p.Mag2(); A(0,1) = p.X()*p.Y()/p.Mag2(); A(1,0) = A(0,1);
  B(0,0) = d.X()*d.X()/(Lxy*Lxy); B(1,1) = d.Y()*d.Y()/(Lxy*Lxy); B(0,1) = d.X()*d.Y()/(Lxy*Lxy); B(1,0) = B(0,1);
  C(0,0) = d.X()*p.X()/(Lxy*p.Mag()); C(1,1) = d.Y()*p.Y()/(Lxy*p.Mag()); C(0,1) = d.X()*p.Y()/(Lxy*p.Mag()); C(1,0) = d.Y()*p.X()/(Lxy*p.Mag());

  TMatrixD EP(VP); EP*= ((double)1.0/p.Mag2());
  TMatrixD EL(VL); EL*= ((double)1.0/(Lxy*Lxy));

  TMatrixD A1(A); A1*=(double)4.0; A1+=B;
  TMatrixD C1(C); C1*=(double)4.0; A1-=C1;
  
  TMatrixD A_EL(A,TMatrixD::kMult,EL);
  TMatrixD A1_EP(A1,TMatrixD::kMult,EP);
  TMatrixD SL = A_EL; SL+=A1_EP;
  double sLxy2 = SL(0,0) + SL(1,1); 
  double ect = (double) fabs(lf)*sqrt(sLxy2);

  std::vector<double> myctv;
  myctv.push_back( ct );
  myctv.push_back( ect );
  return myctv;
}

std::vector<double> MydeltaR( const TVector3 &Mu1, const TVector3 &Mu2, const TLorentzVector genmu1, const TLorentzVector genmu2, 
                              const int &mu_charge1, const int &mu_charge2, const TVector3 &Pi1, const TVector3 &Pi2, 
                              const TLorentzVector genk, const TLorentzVector genpi, const int &pi_charge1, const int &pi_charge2){
    
    // Matching for muons 
    Double_t dr_mu1, dr_mu2;
    if(mu_charge1==1){
      dr_mu1 = deltaR_(Mu1, genmu2.Vect());
      dr_mu2 = deltaR_(Mu2, genmu1.Vect());
    } else {
      dr_mu1 = deltaR_(Mu1, genmu1.Vect());
      dr_mu2 = deltaR_(Mu2, genmu2.Vect());
    }

    // Matching for pions/kaons
    const double m_K = 0.493677;
    const double m_Pi = 0.139570;   

    TLorentzVector Tr_1_K, Tr_2_pi, Tr_1_Pi, Tr_2_K;
    Tr_1_K.SetXYZM(Pi1.X(), Pi1.Y(), Pi1.Z(), m_K);
    Tr_2_pi.SetXYZM(Pi2.X(), Pi2.Y(), Pi2.Z(), m_Pi);
    Tr_1_Pi.SetXYZM(Pi1.X(), Pi1.Y(), Pi1.Z(), m_Pi);
    Tr_2_K.SetXYZM(Pi2.X(), Pi2.Y(), Pi2.Z(), m_K);

    Double_t dr_track1K = deltaR_(Tr_1_K.Vect(), genk.Vect());
    Double_t dr_track2Pi  = deltaR_(Tr_2_pi.Vect(), genpi.Vect());
    Double_t dr_track1Pi  = deltaR_(Tr_1_Pi.Vect(), genpi.Vect());
    Double_t dr_track2K = deltaR_(Tr_2_K.Vect(), genk.Vect());

    std::vector<double> myv;
    myv.push_back(dr_mu1);
    myv.push_back(dr_mu2);
    myv.push_back(dr_track1K);
    myv.push_back(dr_track2Pi);
    myv.push_back(dr_track1Pi);
    myv.push_back(dr_track2K);
    return myv;
}