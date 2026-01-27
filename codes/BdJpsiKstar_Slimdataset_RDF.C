#include <iostream>
#include <string>
#include <TMath.h>
#include <math.h>
#include <Math/Vector4D.h>
#include "Math/GenVector/Boost.h"
#include <vector>
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TMatrixD.h"
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TROOT.h>
#include "RooFit.h"
#include <ROOT/RDataFrame.hxx>
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"

#include "Math/Vector4D.h"
#include "Math/Vector3D.h"


using namespace std;
using namespace ROOT;


////////////////////////////
std::vector<double> ComputeAngles(
    double mu1_px, double mu1_py, double mu1_pz, int mu1_ch,
    double mu2_px, double mu2_py, double mu2_pz, int mu2_ch,
    double trk1_px, double trk1_py, double trk1_pz, int trk1_ch,
    double trk2_px, double trk2_py, double trk2_pz, int trk2_ch,
    int hypothesis // 1: Trk1=Pion, Trk2=Kaon.  2: Trk1=Kaon, Trk2=Pion
);


ROOT::RDF::RNode AddSelection(ROOT::RDF::RNode node,
                              std::string filter1,
                              std::string filter2,
                              std::string filter3, 
                              std::string filter4);


ROOT::RDF::RNode SetVariables(ROOT::RDF::RNode node, 
                              UInt_t WhatSample);

Double_t deltaR_( const TVector3 &v1, 
                  const TVector3 &v2);

std::vector<double> MydeltaR( const TVector3 &Mu1, 
                              const TVector3 &Mu2, 
                              const TLorentzVector genmu1, 
                              const TLorentzVector genmu2, 
                              const int &mu_charge1, 
                              const int &mu_charge2, 
                              const TVector3 &Pi1, 
                              const TVector3 &Pi2, 
                              const TLorentzVector genpi1, 
                              const TLorentzVector genpi2, 
                              const int &pi_charge1, 
                              const int &pi_charge2);

std::vector<double> Lifetime( TVector3 &pv, 
                              TVector3 &sv, 
                              vector<vector<double>> &EPV_i, 
                              vector<vector<double>> &ESV_i, 
                              TVector3 &pT, 
                              Double_t &M) // Note: M("Mass") is now by hand
{
  //NOTE1: This function calculates the lifetime and its error using the transverse proper decay  length.
  //Remember that we are assuming that the error in the pt is negligible, therefore the matrices associated with it are defined as zero

  //Double_t M = 5.279;
  TMatrixD EPV(3,3);
  EPV(0,0) = EPV_i.at(0).at(0);
  EPV(1,1) = EPV_i.at(1).at(1);
  EPV(2,2) = EPV_i.at(2).at(2);
  EPV(0,1) = EPV_i.at(0).at(1);
  EPV(0,2) = EPV_i.at(0).at(2);
  EPV(1,2) = EPV_i.at(1).at(2);  
  
  TMatrixD ESV(3,3);
  ESV(0,0) = ESV_i.at(0).at(0);
  ESV(1,1) = ESV_i.at(1).at(1);
  ESV(2,2) = ESV_i.at(2).at(2);
  ESV(0,1) = ESV_i.at(0).at(1);
  ESV(0,2) = ESV_i.at(0).at(2);
  ESV(1,2) = ESV_i.at(1).at(2);
  /*
   for(int i=0;i<3;i++)
  {
    for(int j=0;j<3;j++)
    {
      EPV[i][j] = EPV_[i][j];
      ESV[i][j] = ESV_[i][j];
    }
  }  
  */
  
  TVector3 svT(sv.X(),sv.Y(),0.0);
  TVector3 pvT(pv.X(),pv.Y(),0.0);
  TVector3 d = svT - pvT;

  TMatrixD VSV(2,2);
  VSV(0,0) = ESV(0,0);
  VSV(1,1) = ESV(1,1);
  VSV(1,0) = ESV(1,0);
  VSV(0,1) = VSV(1,0);

  TMatrixD VPV(2,2);
  VPV(0,0) = EPV(0,0);
  VPV(1,1) = EPV(1,1);
  VPV(1,0) = EPV(1,0);
  VPV(0,1) = VPV(1,0);

  TMatrixD VL(2,2); VL = VSV; VL+=VPV; 
  TVector3 p = pT;

  TMatrixD VP(2,2);
  VP(0,0) = 0.0;
  VP(1,1) = 0.0;
  VP(0,1) = 0.0;
  VP(1,0) = 0.0;  

  double Lxy = d.Dot(p)/p.Mag();
  double lf = Lxy*M/p.Mag();
  //cout<<" ---> "<<lf<<endl;
  double ct =  lf;

  //Now, the lifetime error
  
  //computing Mass error
  //double sM2 = 0; //We assume 0 for now
  
  //computing Lxy error
  
  //Defining Matrix:
  TMatrixD A(2,2);
  TMatrixD B(2,2);
  TMatrixD C(2,2);
  TMatrixD EP(2,2);
  TMatrixD EL(2,2);
  
  //Aij = PiPj/p2
  //Bij = LiLj/Lxy2 (Li = SVi - PVi)
  //EPij = Vij(P)/p2
  //ELij = Vij(L)/Lxy^2;
  //Cij = LiPj/(pLxy)
  
  A(0,0) = p.X()*p.X()/p.Mag2();
  A(1,1) = p.Y()*p.Y()/p.Mag2();
  A(0,1) = p.X()*p.Y()/p.Mag2();
  A(1,0) = A(0,1);

  B(0,0) = d.X()*d.X()/(Lxy*Lxy);
  B(1,1) = d.Y()*d.Y()/(Lxy*Lxy);
  B(0,1) = d.X()*d.Y()/(Lxy*Lxy);
  B(1,0) = B(0,1);
  
  C(0,0) = d.X()*p.X()/(Lxy*p.Mag());
  C(1,1) = d.Y()*p.Y()/(Lxy*p.Mag());
  C(0,1) = d.X()*p.Y()/(Lxy*p.Mag());
  C(1,0) = d.Y()*p.X()/(Lxy*p.Mag());

  EP = VP;
  EP*= ((double)1.0/p.Mag2());
  EL = VL;
  EL*= ((double)1.0/(Lxy*Lxy));

  //Test
  //EL(0,1) = 0.0;
  //EL(1,0) = 0.0;

  //Calculated Sigma Lxy
  // Sigma Lxy^2 = Tr{A*EL + (B + 4*A - 4*C)*EP}
  // NOTA2: in our case it is basically Sigma Lxy^2 = Tr{A*EL), since we do not consider the momentum P
  
  TMatrixD A1 = A;
  A1*=(double)4.0;
  A1+=B;
  TMatrixD C1 = C;
  C1*=(double)4.0;
  A1-=C1;
  
  TMatrixD A_EL(A,TMatrixD::kMult,EL);
  TMatrixD A1_EP(A1,TMatrixD::kMult,EP);
  TMatrixD SL = A_EL;SL+=A1_EP;
  double sLxy2 = SL(0,0) + SL(1,1); 

  //return ct;
  //return ect = (double) fabs(lf)*sqrt(sLxy2);
  
  double ect = (double) fabs(lf)*sqrt(sLxy2);
  std::vector<double> myctv;
  myctv.push_back( ct );
  myctv.push_back( ect );
  //std::cout << "mass = " << M << ", lifetime = " << ct << std::endl;
  //std::cout << " " << std::endl;

  return myctv;
  
}

//"const" maybe we must not give rights to the functions. 
//In the end, the variables will not change, we just want the return 
TVector3 myTV3(const double &x, 
              const double &y, 
              const double &z)
              {TVector3 theV(x,y,z); return theV;  }



TVector3 myTV3_2(const double &x, 
                const double &y, 
                const double &z)
                {TVector3 theV(x,y,0); return theV;  }



vector<vector<double>> myTM(const double &x, 
                            const double &y, 
                            const double &z, 
                            const double &xy, 
                            const double &xz, 
                            const double &yz){
  vector<vector<double>> theM{{0,0,0},{0,0,0},{0,0,0}};
  theM.at(0).at(0) = x;
  theM.at(1).at(1) = y;
  theM.at(2).at(2) = z;
  theM.at(0).at(1) = xy;
  theM.at(0).at(2) = xz;
  theM.at(1).at(2) = yz;
  return theM;
}



std::vector<double> MassVeto(const double &x1,
                             const double &y1,
                             const double &z1, 
                             const double &x2, 
                             const double &y2, 
                             const double &z2)
{
  TLorentzVector pi1,pi2, pipi;
  pi1.SetXYZM(x1, y1, z1, 0.13957);
  pi2.SetXYZM(x2, y2, z2, 0.13957);
  pipi = pi1+pi2;
  Double_t masspipi_veto = pipi.M();

  TLorentzVector k1,k2, kk;
  k1.SetXYZM(x1, y1, z1, 0.493677);
  k2.SetXYZM(x2, y2, z2, 0.493677);
  kk = k1+k2;
  Double_t masskk_veto = kk.M();

  std::vector<double> myMveto;
  myMveto.push_back( masspipi_veto );
  myMveto.push_back( masskk_veto );
  
  return myMveto;
}


std::vector<double> ComputeAngles(
    double mu1_px, double mu1_py, double mu1_pz, int mu1_ch,
    double mu2_px, double mu2_py, double mu2_pz, int mu2_ch,
    double trk1_px, double trk1_py, double trk1_pz, int trk1_ch,
    double trk2_px, double trk2_py, double trk2_pz, int trk2_ch,
    int hypothesis) 
{
    // Masas PDG
    const double m_mu = 0.105658;
    const double m_K  = 0.493677;
    const double m_pi = 0.139570;

    using LV = ROOT::Math::PxPyPzMVector;
    using XYZVector = ROOT::Math::XYZVector;

    // cuadrivectores de los Muones
    LV p_mu1(mu1_px, mu1_py, mu1_pz, m_mu);
    LV p_mu2(mu2_px, mu2_py, mu2_pz, m_mu);

    // Identificar mu+ y mu- (cosThetaL)
    // El paper define theta_l respecto al mu+
    LV p_mu_plus  = (mu1_ch > 0) ? p_mu1 : p_mu2;
    LV p_mu_minus = (mu1_ch > 0) ? p_mu2 : p_mu1;
    LV p_dimuon   = p_mu_plus + p_mu_minus;

    // cuadrivectores de los Hadrones (Según Hipótesis)
    LV p_trk1, p_trk2;
    LV p_K, p_pi;

    // NOTA: hypothesis 1 = piK (Trk1=pi, Trk2=K)
    //       hypothesis 2 = Kpi (Trk1=K, Trk2=pi)
    
    if (hypothesis == 1) {
        p_trk1.SetCoordinates(trk1_px, trk1_py, trk1_pz, m_pi); // Track 1 es Pion
        p_trk2.SetCoordinates(trk2_px, trk2_py, trk2_pz, m_K);  // Track 2 es Kaon
        p_pi = p_trk1;
        p_K  = p_trk2;
    } else {
        p_trk1.SetCoordinates(trk1_px, trk1_py, trk1_pz, m_K);  // Track 1 es Kaon
        p_trk2.SetCoordinates(trk2_px, trk2_py, trk2_pz, m_pi); // Track 2 es Pion
        p_K  = p_trk1;
        p_pi = p_trk2;
    }

    LV p_Kstar = p_K + p_pi;
    LV p_B = p_dimuon + p_Kstar; // Reconstrucción total del B

    // CÁLCULO DE ÁNGULOS
    // CosThetaL (Sistema Dimuon)
    // Ángulo entre mu+ y la dirección opuesta al B en el marco del Dimuon.
    ROOT::Math::Boost boost_to_dimuon(p_dimuon.BoostToCM());
    LV p_mu_plus_dimuon = boost_to_dimuon(p_mu_plus);
    LV p_B_dimuon       = boost_to_dimuon(p_B);
    
    XYZVector v_mu_plus_dimuon = p_mu_plus_dimuon.Vect();
    XYZVector v_B_dimuon       = p_B_dimuon.Vect();
    
    // direction opposite to that of the B0
    // v_mu_plus . (-v_B) / |v_mu_plus| |-v_B|
    double costheta_l = v_mu_plus_dimuon.Dot(-v_B_dimuon) / (v_mu_plus_dimuon.R() * v_B_dimuon.R());

    // CosThetaK (Sistema K*)
    // Ángulo entre el Kaon y la dirección opuesta al B en el marco del K*
    ROOT::Math::Boost boost_to_Kstar(p_Kstar.BoostToCM());
    LV p_K_Kstar = boost_to_Kstar(p_K);
    LV p_B_Kstar = boost_to_Kstar(p_B);

    XYZVector v_K_Kstar = p_K_Kstar.Vect();
    XYZVector v_B_Kstar = p_B_Kstar.Vect();

    // direction of the K+ and the B0 meson (opposite)
    double costheta_k = v_K_Kstar.Dot(-v_B_Kstar) / (v_K_Kstar.R() * v_B_Kstar.R());

    // Phi (Plano vs Plano en el marco del B)
    ROOT::Math::Boost boost_to_B(p_B.BoostToCM());
    LV p_mu_plus_B = boost_to_B(p_mu_plus);
    LV p_mu_minus_B = boost_to_B(p_mu_minus);
    LV p_K_B        = boost_to_B(p_K);
    LV p_pi_B       = boost_to_B(p_pi);
    // Necesitamos ? el K*0 en el marco del B para el signo
    LV p_Kstar_B    = boost_to_B(p_Kstar); 

    XYZVector v_mu_plus_B  = p_mu_plus_B.Vect();
    XYZVector v_mu_minus_B = p_mu_minus_B.Vect();
    XYZVector v_K_B        = p_K_B.Vect();
    XYZVector v_pi_B       = p_pi_B.Vect();
    XYZVector v_Kstar_B    = p_Kstar_B.Vect(); // Dirección del K*

    // Normales a los planos de decaimiento
    // p_mu+ x p_mu-
    XYZVector n_L = v_mu_plus_B.Cross(v_mu_minus_B);
    // p_K x p_pi
    XYZVector n_K = v_K_B.Cross(v_pi_B);

    double phi = std::acos(n_L.Dot(n_K) / (n_L.R() * n_K.R()));

    // Determinación del signo:
    // v_Kstar . (n_L x n_K)
    double sign_check = v_Kstar_B.Dot(n_L.Cross(n_K));
    if (sign_check < 0) phi = -phi;
    return {costheta_l, costheta_k, phi};
}

// *****************************************************************************
// here you will see many "examples" (discusion) about how to work with boost 
// *****************************************************************************
// https://root.cern/doc/v608/classTLorentzVector.html
// https://root-forum.cern.ch/t/how-to-use-boost-in-tlorentzvector/4102
// https://github.com/scikit-hep/vector/issues/134
// https://root.cern/doc/v626/LorentzVectorPage.html
// https://gitlab.cern.ch/hcrottel/bparknanotuplizer/-/blob/master/plugins/BToKMMBuilder.cc#L681
// You can see a differnt version (with tlorentzvector) im the end of this file
//math::XYZTLorentzVector, ROOT::Math::XYZTVector
//ROOT::Math::PxPyPzMVector myTV4(const double &x, const double &y, const double &z, const double &Mass){ROOT::Math::PxPyPzMVector theV(x,y,z,Mass); return theV;  }

//*******************************************************************************************************
// Notabene1: Very Imporntant
// "year"   is 2022 or 20223 
// "sample" is 1 if is Data, 2 if is MCResonante(J/psi), 
//             3 if is MC Resonante(Psi2S),4 if is MC NoResonante(mumu)
// "era"    is 1for 2023C and 2 for 2023D (this is for the Bpix MC differenced)
//*******************************************************************************************************

//******************************************************************************************************
// Notabene2: mass hypothesis  (those variables: "massB1","massB2","masskstar1","masskstar2",)
// hypothesis1 = piK, This means that Track1 is the pion and Track2 is the kaon.
// hypothesis2 = Kpi, This means that Track2 is the pion and Track1 is the kaon.
//******************************************************************************************************


void BdJpsiKstar_Slimdataset_RDF( UInt_t year=2023, 
                                  UInt_t sample=2, 
                                  UInt_t era=1)
{  
  //gErrorIgnoreLevel = 2001;
  //gROOT->ProcessLine( "gErrorIgnoreLevel = 2001;");// Ignore Warnings
  ROOT::EnableImplicitMT(8);// Tell ROOT you want to go parallel
  
  TChain tree("ntuple");
  //test
  //tree.Add("Rootuple_Bstomumuphi_2023_MiniAOD.root/rootuple/ntuple");
  
  if(year==2023 && sample==1){
    //higgs
    //tree.Add("Rootuple_Bd_tomumupipi_2023_MiniAOD_99.root/rootuple/ntuple");

    if(era==1){
      tree.Add("Rootuple_BdtoJpsiKstar_Data2022.root/rootuple/ntuple");
      
    }
    else{
      tree.Add("Data2023/Bdtomumuks0_miniAOD_LowMass_2023_V3/ParkingDoubleMuonLowMass0/Bdtomumuks0_2023_V3_ParkingDoubleMuonLowMass0-Run2023D-22Sep2023_v1-v1/240911_033504/0000/Rootuple_BdtoJpsiKstar_Data2022_1.root/rootuple/ntuple");
    }
    //tree.Add("0000/*.root/rootuple/ntuple");
  }
  else if(year==2023 && sample==2) {
    //tree.Add("Rootuple_BdtoJpsiKstar_Data2022.root/rootuple/ntuple");
  }




  else if (year==2022 && sample==1){
    tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/Analisys2025/BdJpsiKstar_miniAOD_2022Gv1/ParkingDoubleMuonLowMass1/BdJpsiKstar_2022Gv1_ParkingDoubleMuonLowMass1-Run2022G-PromptReco-v1/250627_043446/0000/*.root/rootuple/ntuple");
    cout << " Running for DATA \n";
  }

  else if(year==2022 && sample==2){
    tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/MC2022/MCReco/0000/Rootuple_BdtoJpsiKstar_MC2022_bacht1_*.root/rootuple/ntuple");
    cout << " Running for J/Psi \n";
  }
  
  else if (year==2022 && sample==3){
    tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/Analisys2025BdtoKstarPsi2s_bacht2_myv1/BdtoKstarPsi2s_Psi2sto2Mu_KstartoKPi_MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/BdtoKstarPsi2s_bacht2_myv1/250627_174931/0000/Rootuple_BdtoPsi2SKstar_MC2022_bacht1_*.root/rootuple/ntuple");
    cout << " Running for Psi(2S) \n";
  }

  else if(year==2022 && sample==4){
    tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/DATASETS/Analisys2025BdtomumuKstar_bacht2_myv2/BdtoKstar2Mu_KstartoKPi_MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/BdtomumuKstar_bacht2_myv2/250709_214854/0000/Rootuple_BdtomumuKstar_MC2022_bacht1_*.root/rootuple/ntuple");
    cout << " Running for NoRes\n";

  }

  else {
    std::cout << "No valid file has been try" << std::endl;
    return;
  }
  
  ROOT::RDataFrame dDF(tree);
  //dDF3.Describe().Print();
  auto nentries = dDF.Count();
  cout << " Total entries in Events " << *nentries <<endl;

  // For simplicity, select only events with trigger fire and require kaons with opposite charge
  auto dDF1 = dDF.Filter("tri_DMu4_LM_Displaced==1 || tri_DMu4_3_LM==1", "selTrigger");
  //std::cout << "Cut report:" << std::endl;
  //auto CutsReport1 = dDF1.Report();
  //CutsReport1->Print();
  
  //******************* Lambda function definition ************************************
  //auto myB = [](double x, double y, double z) {TVector3 B(x,y,x); return B;  };
  //auto dDF3 = dDF1.Define("B", myB, {"B1_px", "B1_py", "B1_pz"});
  
  //*******************  Lambda function definition inside "Define"  ******************
  //auto dDF3 = dDF1.Define("B", [](double B1_px, double B1_py, double B1_pz) { TVector3 B(B1_px, B1_py, B1_pz); return B; }, {"B1_px", "B1_py", "B1_pz"});

  //******************* Custom function definition  ***********************************
  Double_t MpdgB = 5.2796;
  Double_t Mpdgv0 = 0.4976;
  Double_t Mpdgmu = 0.105658;

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
    .Define("MpdgB", [&MpdgB] { return MpdgB; }).Define("Mpdgv0", [&Mpdgv0] { return Mpdgv0; })
    .Define("Mpdgmu", [&Mpdgmu] { return Mpdgmu; })
    // B_Trk_px1 corresponde a "Track 1", B_Trk_px2 corresponde a "Track 2"
    .Define("Hyp1_Flag", [](){ return 1; }) 
    .Define("Hyp2_Flag", [](){ return 2; })
    .Define("Angulos_Hyp1", ComputeAngles, {
      "B_J_px1", "B_J_py1", "B_J_pz1", "B_J_charge1",
      "B_J_px2", "B_J_py2", "B_J_pz2", "B_J_charge2",
      "B_Trk_px1", "B_Trk_py1", "B_Trk_pz1", "B_Trk_charge1",
      "B_Trk_px2", "B_Trk_py2", "B_Trk_pz2", "B_Trk_charge2",
      "Hyp1_Flag" // <--- H1
    })
    .Define("Angulos_Hyp2", ComputeAngles, {
      "B_J_px1", "B_J_py1", "B_J_pz1", "B_J_charge1",
      "B_J_px2", "B_J_py2", "B_J_pz2", "B_J_charge2",
      "B_Trk_px1", "B_Trk_py1", "B_Trk_pz1", "B_Trk_charge1",
      "B_Trk_px2", "B_Trk_py2", "B_Trk_pz2", "B_Trk_charge2",
      "Hyp2_Flag" // <--- H2
    });
    
  // note: RDataFrame does not work with TMatrix. So, it is necesario to add vector<vector>. Chek the next link
  // https://root-forum.cern.ch/t/index-out-of-bounds-on-tmatrixdsym-using-rdataframe-in-pyroot/45854
  //Bd Lifetime
  // Reconstruted mass
  //auto dDF4_1 = dDF3.Define("myct", Lifetime, {"pv","sv","EPV","ESV","pT","B1_mass"}).Define("myctks0", Lifetime, {"sv","svks0","ESV","ESVks0","myV0pt","B_Trk_mass"});
  // PDG mass
  auto dDF4 = dDF3.Define("myct", Lifetime, {"pv","sv","EPV","ESV","pT","MpdgB"});

  //Angular distributions: myAngle (remenber: this a vector with both, costhetaKL and cosMuThetaMu)
  //auto dDF4 = dDF4_1.Define("myAng", myAngle, {"mu14v","mu24v","V0v4","B_J_charge1","B_J_charge2"});
  // ... tus defines anteriores ...
  
  // IMPORTANT: we need to save the Jpsi-vertex or at least "cosAlphaXYtmp" and "sigLxyJtmp"
  auto dDF5 = dDF4
    .Define("dxB", "B_DecayVtxX - priVtxX")
    .Define("dyB", "B_DecayVtxY - priVtxY")
    .Define("dxBE", "B_DecayVtxXE")
    .Define("dyBE", "B_DecayVtxYE")
    .Define("sigLxyBtmp", "(dxB*dxB +dyB*dyB)/sqrt( dxB*dxB*dxBE*dxBE + dyB*dyB*dyBE*dyBE )")
    .Define("cosAlphaXYb", "( B1_px*dxB + B1_py*dyB )/( sqrt(dxB*dxB+dyB*dyB)*B.Pt()  )");
  
  // Selection (relaxed cuts) B_TrkTrk_mass1
  auto selDimuon = "mu1.Pt()>=4.0 && mu2.Pt()>=4.0 && fabs(mu1.Eta())<=2.4 && fabs(mu2.Eta())<=2.4 && mu1soft == 1 && mu2soft == 1 && TMath::IsNaN(myct[0])!=1  && TMath::IsNaN(myct[1])!=1";
  auto selJmass = "B_J_mass>=0.2 &&  B_J_mass<=4.8";
  //auto selOthers = "B1_mass<5.7 && B_Prob>0.01 && B_J_Prob>0.01 && B.Pt()>5.0 && pi1.Pt()>0.0 && pi2.Pt()>0.0 && Jpsi.Pt()>=5.0";
  auto selOthers = "B_J_Prob>0.01 && B.Pt()>5.0 && pi1.Pt()>0.95 && pi2.Pt()>0.95 && Jpsi.Pt()>=5.0";
  auto selOthers2 = "B_TrkTrk_mass1>0.5 && B_TrkTrk_mass1<5.0"; //No sense K* mass cuts
  
  // Just to add the filters
  auto dDF6 = AddSelection(dDF5, selDimuon, selJmass, selOthers, selOthers2);

  // If a branch with that name you want is already present in the input TTree/TChain. Use Redefine to force redefinition or DO NOTHIN AND JUST USE IT.
  // EXAMPLE: "run" and "event" already have the names I want, so it is not necessary to define them again.
  /*
  auto seldata = dDF6.Define("massB","B1_mass")
    .Define("masskstar1","B_TrkTrk_mass1").Define("massJ","B_J_mass").Define("cosalfaB","cosAlphaXYb")
    .Define("Bdl","myct[0]").Define("BdlE","myct[1]").Define("Beta","B.Eta()").Define("Bpt","B.Pt()").Define("Jpsipt","Jpsi.Pt()")
    .Define("Pi1pt","pi1.Pt()").Define("Pi2pt","pi2.Pt()").Define("mu1pt","mu1.Pt()").Define("mu2pt","mu2.Pt()").Define("Jprob","B_J_Prob").Define("Bprob","B_Prob");
  */
  auto seldata = SetVariables(dDF6, sample);
  
  //const vector<string> columns  = {"", "", "","","","","","","", "", "", "","","","","","",""};
  vector<string> columns  = {
            "massB1","massB2","masskstar1","masskstar2","massvetopipi","massvetokk","massJ","cosalfaB","Bdl", "BdlE","Beta","Bpt","Jpsipt"
			     ,"Pi1pt","Pi2pt","mu1pt", "mu2pt", "muon_dca", "Jprob","Bprob","Jeta","mu1eta","mu2eta","Pi1eta","Pi2eta","mu1phi","mu2phi"
           ,"Pi1phi","Pi2phi", "event", "run", "lumiblock", "nVtx","priVtxCL", "tri_DMu4_LM_Displaced","tri_DMu4_3_LM","min_dr_trk1_muons"
           ,"min_dr_trk2_muons","min_dpt_trk1_muons","min_dpt_trk2_muons","mu1medium","mu2medium","Pi1charge","Pi2charge","mu1charge"
           ,"mu2charge","mu1pi1DR","mu1pi2DR","mu2pi1DR","mu2pi2DR","Frompv_Trk1","Frompv_Trk2", "CosThetaL_H1", "CosThetaK_H1", "Phi_H1",
           "CosThetaL_H2", "CosThetaK_H2", "Phi_H2"
          };

  vector<string> columnsMC  = {
    "massBGen","BptGen","BetaGen","massJGen","JpsiptGen","mu1ptGen","mu2ptGen"
    ,"masskstarGen","kstarptGen","drmu1","drmu2","pi1ptGen","pi2ptGen","dr_track1K","dr_track2Pi"
    ,"dr_track1Pi","dr_track2K"
  };
  if(sample!=1){ columns.insert(columns.end(), columnsMC.begin(), columnsMC.end());}
  
  //TString ofile = "ntuple_mumuv0_MiniAOD_2023.root";
  //if (year==2022) ofile = "ntuple_mumuv0_MiniAOD_2022.root";
  TString ofile = Form("ntuple_mumukstar_MiniAOD_Year%1i_Sample%1i_Era%1i.root",year,sample,era) ;
  seldata.Snapshot("treeBd", ofile, columns);
}// End of main funcion

//************************************************************
// Other function implementations (see the definitions above)
//************************************************************

ROOT::RDF::RNode SetVariables(ROOT::RDF::RNode node, UInt_t WhatSample){
  if (WhatSample==1){
    return node
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

      // Para Hypothesis 1 (Pi-K)
      .Define("CosThetaL_H1", "Angulos_Hyp1[0]")
      .Define("CosThetaK_H1", "Angulos_Hyp1[1]")
      .Define("Phi_H1",       "Angulos_Hyp1[2]")

      // Para Hypothesis 2 (K-Pi)
      .Define("CosThetaL_H2", "Angulos_Hyp2[0]")
      .Define("CosThetaK_H2", "Angulos_Hyp2[1]")
      .Define("Phi_H2",       "Angulos_Hyp2[2]");
  }
  else{
    return node
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
      .Define("dr_track2K","mudeltavector[5]")
      .Define("Frompv_Trk1","FrompvTrk1")
      .Define("Frompv_Trk2","FrompvTrk2")
      // Dentro de SetVariables...
      // Para Hypothesis 1 (Pi-K)
      .Define("CosThetaL_H1", "Angulos_Hyp1[0]")
      .Define("CosThetaK_H1", "Angulos_Hyp1[1]")
      .Define("Phi_H1",       "Angulos_Hyp1[2]")

      // Para Hypothesis 2 (K-Pi)
      .Define("CosThetaL_H2", "Angulos_Hyp2[0]")
      .Define("CosThetaK_H2", "Angulos_Hyp2[1]")
      .Define("Phi_H2",       "Angulos_Hyp2[2]")
      ;
    
  }
}

ROOT::RDF::RNode AddSelection( ROOT::RDF::RNode node,
                               std::string filter1, 
                               std::string filter2, 
                               std::string filter3, 
                               std::string filter4){
  auto dDFout = node
                .Filter(filter1, "filter1")
                .Filter(filter2, "filter2")
                .Filter(filter3, "filter3")
                .Filter(filter4, "filter4");
  return dDFout;
}


Double_t deltaR_(const TVector3 &v1,
                 const TVector3 &v2){

    Double_t p1 = v1.Phi();
    Double_t p2 = v2.Phi();
    Double_t e1 = v1.Eta();
    Double_t e2 = v2.Eta();
    Double_t dp = abs(p1 - p2);
    if (dp > 3.141592){
       dp -= (2 * 3.14592);
    }  
    return sqrt((e1 - e2) * (e1 - e2) + dp * dp);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<double> MydeltaR( const TVector3 &Mu1, 
                              const TVector3 &Mu2, 
                              const TLorentzVector genmu1, 
                              const TLorentzVector genmu2, 
                              const int &mu_charge1, 
                              const int &mu_charge2, 
                              const TVector3 &Pi1, 
                              const TVector3 &Pi2,                               
                              const TLorentzVector genk, 
                              const TLorentzVector genpi, 
                              const int &pi_charge1, 
                              const int &pi_charge2){
    
    // Matching for muons 
    Double_t dr_mu1, dr_mu2;
    if(mu_charge1==1){
      dr_mu1 = deltaR_(Mu1, genmu2.Vect());
      dr_mu2 = deltaR_(Mu2, genmu1.Vect());
    } else {
      dr_mu1 = deltaR_(Mu1, genmu1.Vect());
      dr_mu2 = deltaR_(Mu2, genmu2.Vect());
    }

    // Matching for pions
    // Constantes PDG
    const double m_K = 0.493677;
    const double m_Pi = 0.139570;   
    const double m_Kstar = 0.89555; 

    TLorentzVector Tr_1_K, Tr_2_pi, Tr_1_Pi, Tr_2_K;
    
    Tr_1_K.SetXYZM(Pi1.X(), Pi1.Y(), Pi1.Z(), m_K);
    Tr_2_pi.SetXYZM(Pi2.X(), Pi2.Y(), Pi2.Z(), m_Pi);

    Tr_1_Pi.SetXYZM(Pi1.X(), Pi1.Y(), Pi1.Z(), m_Pi);
    Tr_2_K.SetXYZM(Pi2.X(), Pi2.Y(), Pi2.Z(), m_K);



    //Calculo de deltaR para las combinaciones de pions y kaones
    Double_t dr_track1K = deltaR_(Tr_1_K.Vect(), genk.Vect());
    Double_t dr_track2Pi  = deltaR_(Tr_2_pi.Vect(), genpi.Vect());
    //Double_t totalDR_A = dr_track1K_genk + dr_track2Pi_genpi; 

    Double_t dr_track1Pi  = deltaR_(Tr_1_Pi.Vect(), genpi.Vect());
    Double_t dr_track2K = deltaR_(Tr_2_K.Vect(), genk.Vect());
    //Double_t totalDR_B = dr_track1Pi_genpi2 + dr_track2K_genpi1;
    // Double_t dr_K, dr_pi, mass_kstar_dif;
    // bool isSwap = false;
    
    // // Decide which combination to use based on the total deltaR
    // if(totalDR_A < totalDR_B){
    //     dr_K = dr_track1K_genk;
    //     dr_pi = dr_track2Pi_genpi;
    //     isSwap = false;
    //     mass_kstar_dif = fabs((Tr_1_K + Tr_2_pi).M() - m_Kstar);

    // } else {
    //     dr_K = dr_track2K_genpi1;
    //     dr_pi = dr_track1Pi_genpi2;
    //     isSwap = true;
    //     mass_kstar_dif = fabs((Tr_1_Pi + Tr_2_K).M() - m_Kstar);
    // }

    
  

    std::vector<double> myv;
    myv.push_back(dr_mu1);
    myv.push_back(dr_mu2);
    myv.push_back(dr_track1K);
    myv.push_back(dr_track2Pi);
    myv.push_back(dr_track1Pi);
    myv.push_back(dr_track2K);
    return myv;
}

