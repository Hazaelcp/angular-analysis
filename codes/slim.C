#include <iostream>
#include <string>
#include <TMath.h>
#include <math.h>
//#include <Math/Vector4D.h>
#include "Math/GenVector/Boost.h"
//#include "Math/Boost.h"
//#include "Math/BoostXYZ.h"
#include "Math/Vector4D.h"

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

//#include <TError.h>
#include <ROOT/RDataFrame.hxx>
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"

using namespace std;
using namespace ROOT;
//using namespace RooFit;

ROOT::RDF::RNode AddSelection(ROOT::RDF::RNode node, std::string filter1, std::string filter2, std::string filter3, std::string filter4, std::string filter5);
ROOT::RDF::RNode AddFilters(ROOT::RDF::RNode node, const std::vector<std::string>& filters, const string& format); 

ROOT::RDF::RNode SetVariables(ROOT::RDF::RNode node, UInt_t WhatSample);

Double_t deltaR_(const TVector3 &v1, const TVector3 &v2);

std::vector<double> MydeltaR(const TVector3 &Mu1, const TVector3 &Mu2, const TLorentzVector genmu1, const TLorentzVector genmu2, const int &mu_charge1, const int &mu_charge2, const TVector3 &Pi1, const TVector3 &Pi2, const  TLorentzVector genpi1, const  TLorentzVector genpi2, const int &pi_charge1, const int &pi_charge2);


//Double_t V0_Lifetime(TVector3 pv, TVector3 sv, TMatrixD EPV, TMatrixD ESV, Double_t M, TVector3 pT, double &ct, double &ect)
std::vector<double> Lifetime(TVector3 &pv, TVector3 &sv, vector<vector<double>> &EPV_i, vector<vector<double>> &ESV_i, TVector3 &pT, Double_t &M) // Note: M("Mass") is now by hand
{
  //NOTE1: This function calculates the lifetime and its error using the transverse proper decay  length.
  //Remember that we are assuming that the error in the pt is negligible, therefore the matrices associated with it are defined as zero

  //Double_t M = 5.3663;
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
  return myctv;
  
}

//"const" maybe we must not give rights to the functions. 
//In the end, the variables will not change, we just want the return 
TVector3 myTV3(double &x, double &y, double &z){TVector3 theV(x,y,z); return theV;  }

TVector3 myTV3_2(double &x, double &y, double &z){TVector3 theV(x,y,0); return theV;  }

vector<vector<double>> myTM(double &x, double &y, double &z, double &xy, double &xz, double &yz){
  vector<vector<double>> theM{{0,0,0},{0,0,0},{0,0,0}};
  theM.at(0).at(0) = x;
  theM.at(1).at(1) = y;
  theM.at(2).at(2) = z;
  theM.at(0).at(1) = xy;
  theM.at(0).at(2) = xz;
  theM.at(1).at(2) = yz;
  return theM;
}

Double_t MassVetoPion(const double &x1, const double &y1, const double &z1, const double &x2, const double &y2, const double &z2){
  TLorentzVector pp,pm, Kpp;
  pp.SetXYZM(x1, y1, z1, 0.4937); // 0.13957 pion mass
  pm.SetXYZM(x2, y2, z2, 0.4937);
  Kpp = pp+pm;
  Double_t massf0_veto = Kpp.M();
  return massf0_veto;
}

Double_t MassVeto2MuPion(const double &x1, const double &y1, const double &z1, const double &m1, const double &x2, const double &y2, const double &z2){
  TLorentzVector mumu, kaon, inv;
  mumu.SetXYZM(x1, y1, z1, m1);
  kaon.SetXYZM(x2, y2, z2, 0.13957); // pion mass 
  inv = mumu+kaon;
  Double_t mass_veto = inv.M();
  return mass_veto;
}

Double_t MassVeto2MuKaon(const double &x1, const double &y1, const double &z1, const double &x2, const double &y2, const double &z2){
  TLorentzVector p1,p2, sum;
  p1.SetXYZM(x1, y1, z1, 0.4937);
  p2.SetXYZM(x2, y2, z2, 0.13957);// kaon mass
  sum = p1+p2;
  Double_t mSqr = sum.M();
  return mSqr;
}

Double_t MassVeto2Mu2Pion(const double &x1, const double &y1, const double &z1,const double &m1 , 
			  const double &x2, const double &y2, const double &z2,
			  const double &x3, const double &y3, const double &z3){
  TLorentzVector p2,p3; 
  TLorentzVector dimu, sum;
  dimu.SetXYZM(x1, y1, z1, m1);
  p2.SetXYZM(x2, y2, z2, 0.4937);// kaon mass
  p3.SetXYZM(x3, y3, z3, 0.13957);// kaon mass
  sum = p2+p3;
  TLorentzVector invMass;
  invMass = dimu + sum; 
  Double_t mSqr = invMass.M();
  return mSqr;
}

ROOT::Math::PxPyPzMVector myTV4(const double &x, const double &y, const double &z, const double &Mass){ROOT::Math::PxPyPzMVector theV(x,y,z,Mass); return theV;  }
std::vector<double> Angles(const ROOT::Math::PxPyPzMVector muon1, const ROOT::Math::PxPyPzMVector muon2, const int &mu_chrg1, const int &mu_chrg2,
			       const ROOT::Math::PxPyPzMVector kaon1, const ROOT::Math::PxPyPzMVector kaon2, const int &k_chrg1, const int &k_chrg2){

  
  using LV = ROOT::Math::PxPyPzMVector;
  using ROOT::Math::Boost;
  using ROOT::Math::XYZVector;

  // Assign muons and kaons by charge
  LV muplus  = (mu_chrg1 > 0) ? LV(muon1) : LV(muon2);
  LV muminus = (mu_chrg1 > 0) ? LV(muon2) : LV(muon1);
  LV kplus   = (k_chrg1 > 0)  ? LV(kaon1) : LV(kaon2);
  LV kminus  = (k_chrg1 > 0)  ? LV(kaon2) : LV(kaon1);

  LV dilep = muminus + muplus;
  LV dikaon = kplus + kminus;
  LV bs = dilep + dikaon;

  // Angle theta_l 
  Boost boostDiLep(dilep.BoostToCM());
  LV muminus_cm  = boostDiLep(muminus);
  LV muplus_cm  = boostDiLep(muplus);
  LV bs_mucm  = boostDiLep(bs);
  XYZVector v3_muminus_cm = muminus_cm.Vect();
  XYZVector v3_muplus_cm = muplus_cm.Vect();
  XYZVector v3_bs_mucm = bs_mucm.Vect();

  double costheta_l = v3_muminus_cm.Dot(v3_bs_mucm)/(v3_muminus_cm.R() * v3_bs_mucm.R());
  

  // Angle theta_k 

  Boost boostDiKs(dikaon.BoostToCM());
  LV kminus_cm  = boostDiKs(kminus);
  LV kplus_cm  = boostDiKs(kplus);
  LV bs_kcm  = boostDiKs(bs);
  XYZVector v3_kminus_cm = kminus_cm.Vect();
  XYZVector v3_kplus_cm = kplus_cm.Vect();
  XYZVector v3_bs_kcm = bs_kcm.Vect();
  
  double costheta_k = v3_kminus_cm.Dot(v3_bs_kcm)/(v3_kminus_cm.R() * v3_bs_kcm.R());
  //double costheta_ktest = v3_kminus_cm.Dot(v3_kplus_cm)/(v3_kplus_cm.R() * v3_kminus_cm.R());
  //std::cout << ", cosThetaKtest = " << costheta_ktest << std::endl;


  // Angle phi
  Boost boostToBs( bs.BoostToCM());
  LV muplus_bs 	= boostToBs(muplus);  
  LV muminus_bs = boostToBs(muminus);
  LV kplus_bs 	= boostToBs(kplus);
  LV kminus_bs 	= boostToBs(kminus);
  LV dilep_bs 	= boostToBs(dilep);
  LV dikaon_bs 	= boostToBs(dikaon);

  XYZVector v3_muminus_bs = muminus_bs.Vect();
  XYZVector v3_muplus_bs = muplus_bs.Vect();
  XYZVector v3_kminus_bs = kminus_bs.Vect();
  XYZVector v3_kplus_bs = kplus_bs.Vect();
  XYZVector v3_dilep_bs = dilep_bs.Vect();
  XYZVector v3_dikaon_bs = dikaon_bs.Vect();
  XYZVector v3_bs = bs.Vect();

  // Build the planes(vectores normales)
  XYZVector n_l = v3_muplus_bs.Cross(v3_muminus_bs);
  XYZVector n_k = v3_kplus_bs.Cross(v3_kminus_bs);
  
  double phi = std::acos(n_l.Dot(n_k)/(n_l.R() * n_k.R()) );
  double sign = v3_dilep_bs.Dot(n_l.Cross(n_k));
  if (sign < 0)phi = -phi;
 
  std::vector<double> myangv;
  myangv.push_back( costheta_l);
  myangv.push_back( costheta_k);
  myangv.push_back( phi);
  return myangv;
}




//****************************************************************************************************************************
// NB: Very Important
// "year" is 2022 or 2023 
// "sample" is 0/1 if is Data pre/post EE,BPix, 
// 	       2/3 if is MC No-Resonante(mumu) noCuts or PHSP, 	pre/post EE,BPix, 
// 	       4/5 if is MC Resonante(J/psi) noCuts or PHSP, 	pre/post EE,BPix, 
//             6/7 if is MC Resonante(Psi2S) noCuts or PHSP, 	pre/post EE,BPix,
//             8/9 if is MC No-Resonante(mumu) sllBall,		pre/post EE,BPix,
//             10/11 if is MC Resonante(J/psi) SVS, 		pre/post EE,BPix,
//             12/13 if is MC Resonante(Psi2S) SVS,		pre/post EE,BPix.
//****************************************************************************************************************************
void Bsmumuphi_Slimdataset_RDF(UInt_t year=2022, UInt_t sample=0, const string & format ="MINIAOD")
{  
  ROOT::EnableImplicitMT(8);// Tell ROOT you want to go parallel
  
  TChain tree("ntuple");
  
  TString outpath = Form("/cms/data/rreyesal/Bphysics/Bstomumuphi_%i/",year);
  TString path = Form("root://eosuser.cern.ch:///eos/user/r/rreyesal/Bphysics/Bstomumuphi_%i/",year);
  
  string era = "Summer23_preBPix"; 
  if(sample != 0 && sample%2 == 0  ){ 
	  if (year == 2023) era = era;
	  else era = "Summer22FilterFix_preEE";
	  } 
  else if (sample != 1 && sample%2 ){ 
	  if (year == 2023) era = "Summer23_postBPix";
	  else era = "Summer22FilterFix_postEE";
        }
  else if (sample == 0 || sample == 1) { 
	  if (year == 2023 && sample ==0 ) era = Form("Run%iC",year);
	  else if (year == 2023 && sample == 1 ) era = Form("Run%iD",year);
	  else if (year == 2022 && sample ==0 ) era = Form("Run%iCD",year);
	  else if (year == 2022 && sample == 1  ) era = Form("Run%iEFG",year);
  }

  if (year==2023 && sample==0 ){
    //
	std::cout << "era: " << era << endl; 
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
  }
  else if (year==2023 && sample==1 ){
	std::cout << "era: " << era << endl; 
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
  }
  else if (year==2022 && sample==0 ){
	std::cout << "era: " << era << endl; 
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iC_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iD_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
  }
  else if (year==2022 && sample==1 ){   
	std::cout << "era: " << era << endl; 
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass0-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass1-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass2-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass3-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass4-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass5-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass6-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iE_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iF_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
		    tree.Add(path+Form("ParkingDoubleMuonLowMass7-Run%iG_Bstomumuphi_MINIAOD.root/rootuple/ntuple",year));
  }
  else if( sample != 0 && sample%2 ==0 ){
	std::cout << " era: " << era << endl; 
	if( sample == 2)	tree.Add(path+Form("BsTomumuphi_unbiased_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if(sample == 4 ) 	tree.Add(path+Form("BsTojpsiphi_unbiased_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if(sample == 6 )	tree.Add(path+Form("BsTopsiphi_unbiased_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if(sample == 8)	tree.Add(path+Form("BsTomumuphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if(sample ==10) 	tree.Add(path+Form("BsTojpsiphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	//else if(sample ==10) 	tree.Add(path+Form("BTojpsiKstar_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	//else 			tree.Add(path+Form("BTopsiKstar_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if(sample ==12)    tree.Add(path+Form("BsTopsiphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
  
  }
  else if( sample !=1 && sample%2){
	std::cout << " era: " << era << endl; 
	if( sample == 3)	tree.Add(path+Form("BsTomumuphi_unbiased_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	//else if( sample == 5) 	tree.Add(path+Form("BsTojpsiphi_MuFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if( sample == 5) 	tree.Add(path+Form("BsTojpsiphi_unbiased_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if( sample == 7)	tree.Add(path+Form("BsTopsiphi_unbiased_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if( sample == 9 && year==2023)	tree.Add(path+Form("BsTomumuphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if( sample == 9 && year==2022)	tree.Add(path+Form("BsTomumuphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else if( sample == 11) 	tree.Add(path+Form("BsTojpsiphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	else 			tree.Add(path+Form("BsTopsiphi_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	//else if( sample == 11) 	tree.Add(path+Form("BTojpsiKstar_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
	//else 			tree.Add(path+Form("BTopsiKstar_muFilter_MCRun3%s_%s.root/rootuple/ntuple",era.c_str(), format.c_str() ));
  } 


  ROOT::RDataFrame dDF(tree);
  //dDF3.Describe().Print();
  auto nentries = dDF.Count();
  cout << " Total entries in Events " << *nentries <<endl;

  // For simplicity, select only events with trigger fire and require kaons with opposite charge
  
  //******************* Lambda function definition ************************************
  //auto myB = [](double x, double y, double z) {TVector3 B(x,y,x); return B;  };
  //auto dDF3 = dDF2.Define("B", myB, {"B_px", "B_py", "B_pz"});
  
  //*******************  Lambda function definition inside "Define"  ******************
  //auto dDF3 = dDF2.Define("B", [](double B_px, double B_py, double B_pz) { TVector3 B(B_px, B_py, B_pz); return B; }, {"B_px", "B_py", "B_pz"});

  //******************* Custom function definition  ***********************************
  Double_t MpdgB = 5.3663;
  Double_t Mpdgmu = 0.105658;
  Double_t Mkaon = 0.4937;
  Double_t Mpion = 0.13957;


  auto dDF3 = dDF.Define("B", myTV3, {"B_px", "B_py", "B_pz"}).Define("Jpsi", myTV3, {"B_J_px", "B_J_py", "B_J_pz"})
  //auto dDF3 = dDF2.Define("B", myTV3, {"B_px", "B_py", "B_pz"}).Define("Jpsi", myTV3, {"B_J_px", "B_J_py", "B_J_pz"})
    .Define("mu1", myTV3, {"B_J_px1", "B_J_py1", "B_J_pz1"}).Define("mu2", myTV3, {"B_J_px2", "B_J_py2", "B_J_pz2"})
    .Define("K1", myTV3, {"B_phi_px1", "B_phi_py1", "B_phi_pz1"})
    .Define("K2", myTV3, {"B_phi_px2", "B_phi_py2", "B_phi_pz2"})
    .Define("DRmu1trk1",deltaR_, {"mu1", "K1"})
    .Define("DRmu2trk2",deltaR_, {"mu2", "K2"})
    .Define("pT", myTV3_2, {"B_px", "B_py", "B_pz"})
    .Define("pv", myTV3, {"priVtxX", "priVtxY", "priVtxZ"}).Define("sv", myTV3, {"B_DecayVtxX", "B_DecayVtxY", "B_DecayVtxZ"})
    .Define("EPV", myTM, {"priVtxXE", "priVtxYE", "priVtxZE", "priVtxXYE", "priVtxXZE", "priVtxYZE"})
    .Define("ESV", myTM, {"B_DecayVtxXE", "B_DecayVtxYE", "B_DecayVtxZE", "B_DecayVtxXYE", "B_DecayVtxXZE", "B_DecayVtxYZE"})
    .Define("massveto", MassVetoPion, {"B_phi_px1", "B_phi_py1", "B_phi_pz1","B_phi_px2", "B_phi_py2", "B_phi_pz2"})
    .Define("mumuK1",  MassVeto2MuKaon, {"B_phi_px1", "B_phi_py1", "B_phi_pz1","B_phi_px2", "B_phi_py2", "B_phi_pz2"})
    .Define("mumuK2",  MassVeto2MuKaon, {"B_phi_px2", "B_phi_py2", "B_phi_pz2","B_phi_px1", "B_phi_py1", "B_phi_pz1"})
    .Define("mumuK0_ld", MassVeto2Mu2Pion, {"B_J_px", "B_J_py", "B_J_pz","B_J_mass","B_phi_px1_track", "B_phi_py1_track", "B_phi_pz1_track",
    					  "B_phi_px2_track", "B_phi_py2_track", "B_phi_pz2_track"})
    .Define("mumuK0_tr", MassVeto2Mu2Pion, {"B_J_px", "B_J_py", "B_J_pz","B_J_mass","B_phi_px2_track", "B_phi_py2_track", "B_phi_pz2_track",
    					  "B_phi_px1_track", "B_phi_py1_track", "B_phi_pz1_track"})
    .Define("MpdgB", [&MpdgB] { return MpdgB; }).Define("Mkaon", [&Mkaon] { return Mkaon; }).Define("Mpdgmu", [&Mpdgmu] { return Mpdgmu; })
    .Define("mu14v", myTV4, {"B_J_px1", "B_J_py1", "B_J_pz1","Mpdgmu"}).Define("mu24v", myTV4, {"B_J_px2", "B_J_py2", "B_J_pz2","Mpdgmu"})
    .Define("K14v", myTV4, {"B_phi_px1", "B_phi_py1", "B_phi_pz1","Mkaon"}).Define("K24v", myTV4, {"B_phi_px2", "B_phi_py2", "B_phi_pz2","Mkaon"})
    .Define("angles",Angles,{"mu14v","mu24v","B_J_charge1","B_J_charge2","K14v","K24v","B_phi_charge1","B_phi_charge2"}); 
  
  // note: RDataFrame does not work with TMatrix. So, it is necesario to add vector<vector>. Chek the next link
  // https://root-forum.cern.ch/t/index-out-of-bounds-on-tmatrixdsym-using-rdataframe-in-pyroot/45854
  auto dDF4 = dDF3.Define("myct", Lifetime, {"pv","sv","EPV","ESV","pT","MpdgB"});

  // IMPORTANT: we need to save the Jpsi-vertex
  auto dDF5 = dDF4.Define("dxB", "B_DecayVtxX - priVtxX").Define("dyB", "B_DecayVtxY - priVtxY").Define("dxBE", "B_DecayVtxXE").Define("dyBE", "B_DecayVtxYE")
    .Define("sigLxyBtmp", "(dxB*dxB +dyB*dyB)/sqrt( dxB*dxB*dxBE*dxBE + dyB*dyB*dyBE*dyBE )")
    .Define("cosAlphaXYb", "( B_px*dxB + B_py*dyB )/( sqrt(dxB*dxB+dyB*dyB)*B.Pt()  )");


  std::vector<std::string> myFilters = {
  	"B_phi_charge1!=B_phi_charge2",
	"tri_DMu4_LM_Displaced==1 || tri_DMu4_3_LM==1", 
	"mu1.Pt()>=4.0 && mu2.Pt()>=4.0",
	"fabs(mu1.Eta())<=2.4 && fabs(mu2.Eta())<=2.4",
	"mu1medium == 1 && mu2medium == 1",
	"TMath::IsNaN(myct[0])!=1  && TMath::IsNaN(myct[1])!=1", 
	"B_J_mass>=0.2 &&  B_J_mass<=4.8", 
	"B_mass<5.7", 
	"B_J_Prob>0.01 && B_Prob>0.01",
	"K1.Pt()>0.95 && K2.Pt()>0.95", 
	"Jpsi.Pt()>=5.0",
	"B.Pt()>5.0", 
	"myct[0]>0.0", 
	"B_phi_mass>(1.01946-0.020) && B_phi_mass<(1.01946+0.020)",
  };
  auto dDF6 = AddFilters(dDF5, myFilters, format);
  //std::cout << "Cut report:" << std::endl;
  //auto CutsReport1 = dDF1.Report();
  //auto CutsReport2 = dDF2.Report();
  //CutsReport1->Print();
  //CutsReport2->Print();

  
  //Add the filters
  // Just to learn to use ROOT::RDF::RNode. See examples:
  //https://root.cern/doc/master/df025__RNode_8C.html
  //https://root.cern.ch/doc/master/df103__NanoAODHiggsAnalysis_8C.html
  //auto dDF6 = AddSelection(dDF5, selDimuon, selDrTrkMuon ,selJmass, selOthers, selOthers2);

  // If a branch with that name you want is already present in the input TTree/TChain. Use Redefine to force redefinition or DO NOTHIN AND JUST USE IT.
  // EXAMPLE: "run" and "event" already have the names I want, so it is not necessary to define them again.
  /*
  auto seldata = dDF6.Define("triJtrk","tri_JpsiTk").Define("massB","B_mass")
    .Define("massphi","B_phi_mass").Define("massJ","B_J_mass").Define("massJErr","B_J_massErr").Define("sigLxyB","sigLxyBtmp").Define("cosalfaB","cosAlphaXYb")
    .Define("Bdl","myct[0]").Define("BdlE","myct[1]").Define("Beta","B.Eta()").Define("Bpt","B.Pt()").Define("Jpsipt","Jpsi.Pt()")
    .Define("K1pt","K1.Pt()").Define("K2pt","K2.Pt()").Define("mu1pt","mu1.Pt()").Define("mu2pt","mu2.Pt()").Define("Jprob","B_J_Prob").Define("Bprob","B_Prob");
  */
  auto  seldata = SetVariables(dDF6, sample); 


  vector<string> columns  = { "massB","massphi","massveto","massJ","massJErr","q2","sigLxyB","mumuK1","mumuK2","mumuK0_ld","mumuK0_tr","cosalfaB"
		              ,"Bdl", "BdlE","Beta","Bpt","Jpsipt","K1pt","K2pt","mu1pt", "mu2pt", "muon_dca","Jprob","Bprob"
			      ,"Jeta","mu1eta","mu2eta","K1eta","K2eta"
			      ,"DRmu1trk1","DRmu2trk2"
			      ,"min_dr_trk1_muons", "min_dr_trk2_muons"
			      ,"costhetal","costhetak","phi"
			      ,"event", "run", "priVtxCL", "tri_DMu4_LM_Displaced","tri_DMu4_3_LM"};

  vector<string> columnsMC  = {"massBGen","BptGen","BetaGen","massJGen","JpsiptGen",
	  			"mu1ptGen","mu2ptGen","mu1etaGen","mu2etaGen"
  			       ,"massphiGen","phiptGen","drmu1","drmu2"
			       ,"K1ptGen","K2ptGen", "K1etaGen","K2etaGen"
			       ,"costhetalGen","costhetakGen","phiGen"
			       ,"drpi1","drpi2"};
  if(sample > 1 ){ columns.insert(columns.end(), columnsMC.begin(), columnsMC.end());}

  TString ofile = outpath+Form("ntuple_mumuphi_FFyear%1i_sample%1i_%s.root",year,sample, format.c_str()) ;
  //TString ofile = Form("../data/Kstar_samples/ntuple_mumukpi_year%1i_sample%1i.root",year,sample) ;
  seldata.Snapshot("treeBs", ofile, columns);
  return;
  
}

ROOT::RDF::RNode AddFilters(ROOT::RDF::RNode node, const std::vector<std::string>& filters, const string& format){
    
    if (format.c_str() != TString("GEN")){
    	for (size_t i = 0; i < filters.size(); ++i) {
        	node = node.Filter(filters[i], "filter" + std::to_string(i));
    		}
    	return node;
  	}
    else return node; 
} 
ROOT::RDF::RNode AddSelection(ROOT::RDF::RNode node, std::string filter1, std::string filter2, std::string filter3, std::string filter4, std::string filter5)
{
  auto dDFout = node.Filter(filter1, "filter1").Filter(filter2, "filter2").Filter(filter3, "filter3").Filter(filter4, "filter4").Filter(filter5, "filter5");
  return dDFout;
}



ROOT::RDF::RNode SetVariables(ROOT::RDF::RNode node, UInt_t WhatSample){
  if (WhatSample==1 || WhatSample == 0){
    return node.Define("massB","B_mass")
      .Define("massphi","B_phi_mass").Define("massJ","B_J_mass").Define("q2","B_J_mass*B_J_mass").Define("massJErr","B_J_massErr").Define("sigLxyB","sigLxyBtmp").Define("cosalfaB","cosAlphaXYb")
      .Define("Bdl","myct[0]").Define("BdlE","myct[1]").Define("Beta","B.Eta()").Define("Bpt","B.Pt()")
      .Define("K1pt","K1.Pt()").Define("K2pt","K2.Pt()")
      .Define("Jpsipt","Jpsi.Pt()").Define("mu1pt","mu1.Pt()").Define("mu2pt","mu2.Pt()")
      .Define("Jprob","B_J_Prob").Define("Bprob","B_Prob")
      .Define("Jeta","Jpsi.Eta()").Define("mu1eta","mu1.Eta()")
      .Define("mu2eta","mu2.Eta()").Define("K1eta","K1.Eta()")
      .Define("K2eta","K2.Eta()")
      .Define("costhetal","angles[0]").Define("costhetak","angles[1]").Define("phi","angles[2]");
  }
  else{
    Int_t chrgPGen = 1; 
    Int_t chrgMGen = -1; 
    return node.Define("massB","B_mass")
      .Define("massphi","B_phi_mass").Define("massJ","B_J_mass").Define("q2","B_J_mass*B_J_mass").Define("massJErr","B_J_massErr").Define("sigLxyB","sigLxyBtmp").Define("cosalfaB","cosAlphaXYb")
      .Define("Bdl","myct[0]").Define("BdlE","myct[1]").Define("Beta","B.Eta()").Define("Bpt","B.Pt()")
      .Define("K1pt","K1.Pt()").Define("K2pt","K2.Pt()")
      .Define("Jpsipt","Jpsi.Pt()").Define("mu1pt","mu1.Pt()").Define("mu2pt","mu2.Pt()")
      .Define("Jprob","B_J_Prob").Define("Bprob","B_Prob")
      .Define("Jeta","Jpsi.Eta()").Define("mu1eta","mu1.Eta()").Define("mu2eta","mu2.Eta()").Define("K1eta","K1.Eta()").Define("K2eta","K2.Eta()")
      .Define("costhetal","angles[0]")
      .Define("costhetak","angles[1]").Define("phi","angles[2]")
      .Define("massBGen","gen_b_p4.M()").Define("BptGen","gen_b_p4.Pt()").Define("BetaGen","gen_b_p4.Rapidity()")
      .Define("massJGen","gen_jpsi_p4.M()").Define("JpsiptGen","gen_jpsi_p4.Pt()")
      .Define("mu1ptGen","gen_muon1_p4.Pt()").Define("mu2ptGen","gen_muon2_p4.Pt()")
      .Define("mu1etaGen","gen_muon1_p4.Eta()").Define("mu2etaGen","gen_muon2_p4.Eta()")
      .Define("mu1pxGen","gen_muon1_p4.Px()").Define("mu1pyGen","gen_muon1_p4.Py()").Define("mu1pzGen","gen_muon1_p4.Pz()").Define("mu1MGen","gen_muon1_p4.M()")
      .Define("mu2pxGen","gen_muon2_p4.Px()").Define("mu2pyGen","gen_muon2_p4.Py()").Define("mu2pzGen","gen_muon2_p4.Pz()").Define("mu2MGen","gen_muon2_p4.M()")
      .Define("k1pxGen","gen_kaon1_p4.Px()").Define("k1pyGen","gen_kaon1_p4.Py()").Define("k1pzGen","gen_kaon1_p4.Pz()").Define("k1MGen","gen_kaon1_p4.M()")
      .Define("k2pxGen","gen_kaon2_p4.Px()").Define("k2pyGen","gen_kaon2_p4.Py()").Define("k2pzGen","gen_kaon2_p4.Pz()").Define("k2MGen","gen_kaon2_p4.M()")
      .Define("massphiGen","gen_phi_p4.M()").Define("phiptGen","gen_phi_p4.Pt()")
      .Define("K1ptGen","gen_kaon1_p4.Pt()").Define("K2ptGen","gen_kaon2_p4.Pt()")
      .Define("K1etaGen","gen_kaon1_p4.Eta()").Define("K2etaGen","gen_kaon2_p4.Eta()")
      .Define("mudeltavector", MydeltaR, {"mu1", "mu2", "gen_muon1_p4","gen_muon2_p4","B_J_charge1","B_J_charge2","K1", "K2", "gen_kaon1_p4","gen_kaon2_p4","B_phi_charge1","B_phi_charge2"})
      .Define("mu1Gen4", myTV4, {"mu1pxGen", "mu1pyGen", "mu1pzGen","mu1MGen"})
      .Define("mu2Gen4", myTV4, {"mu2pxGen", "mu2pyGen", "mu2pzGen","mu2MGen"})
      .Define("k1Gen4", myTV4, {"k1pxGen", "k1pyGen", "k1pzGen","k1MGen"})
      .Define("k2Gen4", myTV4, {"k2pxGen", "k2pyGen", "k2pzGen","k2MGen"})
      .Define("chrgPGen", [&chrgPGen] { return chrgPGen; }).Define("chrgMGen", [&chrgMGen] { return chrgMGen; })
      .Define("anglesGen",Angles,{"mu1Gen4","mu2Gen4","chrgMGen","chrgPGen","k1Gen4","k2Gen4","chrgPGen","chrgMGen"})
      .Define("costhetalGen","anglesGen[0]").Define("costhetakGen","anglesGen[1]").Define("phiGen","anglesGen[2]")
      .Define("drmu1","mudeltavector[0]").Define("drmu2","mudeltavector[1]").Define("drpi1","mudeltavector[2]").Define("drpi2","mudeltavector[3]");
  }
}

Double_t deltaR_(const TVector3 &v1, const TVector3 &v2){
    Double_t p1 = v1.Phi();
    Double_t p2 = v2.Phi();
    Double_t e1 = v1.Eta();
    Double_t e2 = v2.Eta();
    Double_t dp = abs(
        p1 - p2);
    if (dp > 3.141592){
       dp -= (2 * 3.14592);
    }  
    return sqrt((e1 - e2) * (e1 - e2) + dp * dp);
};

std::vector<double> MydeltaR(const TVector3 &Mu1, const TVector3 &Mu2, const TLorentzVector genmu1, const  TLorentzVector genmu2, const int &mu_charge1, const int &mu_charge2, const TVector3 &Pi1, const TVector3 &Pi2, const TLorentzVector genpi1, const TLorentzVector genpi2, const int &pi_charge1, const int &pi_charge2){
  Double_t dr_mu1, dr_mu2;
  if(mu_charge1==1){
    dr_mu1 = deltaR_(Mu1, genmu2.Vect());//genmu1 is the negative one (mu-==13, mu+==-13)
    dr_mu2 = deltaR_(Mu2, genmu1.Vect());
  }
  else {
    dr_mu1 = deltaR_(Mu1, genmu1.Vect());
    dr_mu2 = deltaR_(Mu2, genmu2.Vect());
  }

  Double_t dr_pi1, dr_pi2;
  if(pi_charge1==1){
    dr_pi1 = deltaR_(Pi1, genpi1.Vect());//genpi1 is the positive one (k+==321, k-==-321)
    dr_pi2 = deltaR_(Pi2, genpi2.Vect());
  }
  else {
    dr_pi1 = deltaR_(Pi1, genpi2.Vect());
    dr_pi2 = deltaR_(Pi2, genpi1.Vect());
  }

  std::vector<double> myv;
  myv.push_back( dr_mu1 );
  myv.push_back( dr_mu2 );
  myv.push_back( dr_pi1 );
  myv.push_back( dr_pi2 );
  //std::cout << "dr_mu1 = " << dr_mu1 << ", dr_mu2 = " << dr_mu2 << std::endl;
  //std::cout << "dr_pi1 = " << dr_pi1 << ", dr_pi2 = " << dr_pi2 << std::endl;
  //std::cout << " " << std::endl;
  return myv;
}