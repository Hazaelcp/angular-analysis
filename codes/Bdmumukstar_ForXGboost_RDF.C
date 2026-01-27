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
  
//#include <TError.h>
#include <ROOT/RDataFrame.hxx>
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"

using namespace std;
using namespace ROOT;
//using namespace RooFit;

ROOT::RDF::RNode AddSelection(ROOT::RDF::RNode node, 
                              std::string filter1, 
                              std::string filter2, 
                              std::string filter3, 
                              std::string filter4, 
                              std::string filter5, 
                              std::string filter6, 
                              std::string filter7, 
                              std::string filter8, 
                              std::string filter9,
                              //std::string filter10, 
                              std::string filtermctching, 
                              UInt_t WhatSample,  
                              UInt_t WhatVersion)
{
  
  if (WhatSample==1 || WhatSample==5){
    
    if(WhatVersion==1){
      cout << " DATA Antiradiation vetos " << endl;
      // return node.Filter(filter1, "filter1")
      //           .Filter(filter2, "filter2")
      //           .Filter(filter3, "filter3")
      //           .Filter(filter4, "filter4")
      //           .Filter(filter5, "filter5")
      //           .Filter(filter6, "filter6")
      //           .Filter(filter7, "filter7");

      // FILTROS APLICADOS PARA LA PRUEBA LOWMASS1G2022
      return node.Filter(filter1, "filter1") //selTrigger
                .Filter(filter2, "filter2")  //selVetoPhi
                .Filter(filter3, "filter3")  //selJvetomass
                .Filter(filter4, "filter4")  // antiradiation jpsi
                .Filter(filter6, "filter6")  // antiradiation psi2s
                .Filter(filter8, "filter8")  //selvetokstar
                .Filter(filter9, "filter9")  //selvetoBmass
                //.Filter(filter10,"filter10");//selNominal
                ;
    }
    else{
      cout << " DATA NO antiradiation vetos " << endl;
      return node.Filter(filter1, "filter1")
                .Filter(filter2, "filter1")
                .Filter(filter8, "filter8")
                ;
    }

  }


  else{  
    if(WhatVersion==1){
      cout << " MC Antiradiation vetos " << endl;
    // return node.Filter(filter1, "filter1")
    //           .Filter(filter2, "filter2")
    //           .Filter(filter3, "filter3")
    //           .Filter(filter4, "filter4")
    //           .Filter(filter5, "filter5")
    //           .Filter(filter6, "filter6")
    //           .Filter(filter7, "filter7")
    //           .Filter(filtermctching, "filtermctching");
    return node.Filter(filter1, "filter1")
              .Filter(filter2, "filter2")
              .Filter(filter3, "filter3")
              .Filter(filter4, "filter4")
              .Filter(filter6, "filter6")
              .Filter(filter8, "filter8")
              //.Filter(filter9, "filter9")
              .Filter(filtermctching, "filtermctching");    

    }
    else{
      cout << " MC NO antiradiation vetos" << endl;
      return node.Filter(filter1, "filter1") // selTrigger
                .Filter(filter2, "filter2")  // selVetoPhi
                .Filter(filter8, "filter8")  // selvetokstar
                .Filter(filter9, "filter9")  // selvetoBmass window
                .Filter(filtermctching, "filtermctching"); //selMCmatching
    } 
  }
  //auto dDFout = node.Filter(filter1, "filter1").Filter(filter2, "filter2").Filter(filter3, "filter3").Filter(filter4, "filter4");
  //return dDFout;
}





//****************************************************************************************************************************
// Notabene: Very Imporntant

// "year"     is 2022 or 2023 
// "sample"   is 1 if is Data, 2 if is MC
// "channel"  is 1 if JPsi, 2 if Psi2S and 3 if NonRes
// "Era"      is 1 or 2 for 2023 data
// "version"  is 1 if used Antidiation vetos, 0 if do NOT used Antidiation vetos
//****************************************************************************************************************************

void Bdmumukstar_ForXGboost_RDF(UInt_t year=2023, UInt_t sample=1, UInt_t channel=0, UInt_t Era=1,  UInt_t version=1)
{  
  std::cout << " year = " << year << ", sample = " << sample 
           << ", channel = " << channel << ", Era = " << Era
          << ", version = " << version << std::endl;

  //gErrorIgnoreLevel = 2001;
  //gROOT->ProcessLine( "gErrorIgnoreLevel = 2001;");// Ignore Warnings
  ROOT::EnableImplicitMT(30);// Tell ROOT you want to go parallel
  
  TChain tree("ntuple");
  //test
  //tree.Add("ntuple_mumuv0_MiniAOD_Year2022_Sample3.root/treeBd");

  // TString basePath = "/home/erickgr/Documents/Particulas/2025/AnalysisButoKstarmumu/Preselection/";
  TString fileName;
  TString channel_name;
  if      (channel == 1) channel_name = "Jpsi";
  else if (channel == 2) channel_name = "Psi2s"; 
  else if (channel == 3) channel_name = "NoRes";

  TString sample_name;
  if      (sample == 1) sample_name = "Data";
  else if (sample == 2) sample_name = "MC";

  
  if (sample==1){
    //tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/CODES/ntuple_mumukstar_MiniAOD_Year2022_Sample1_Era1/*.root/treeBd");
    //tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/CODES/ntuple_mumukstar_MiniAOD_Year2022_Sample1_Era1_low_m1.root/treeBd");
    cout << " Running for DATA \n";
  }
  else if(sample==2 && channel == 1){
    tree.Add("/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/angular/ntuple_mumukstar_MiniAOD_Year2022_Sample2_Era1.root/treeBd");
    cout << " Running for MC J/Psi \n";
  }
  else if (sample==2 && channel == 2){
    tree.Add("/treeBd");
    cout << " Running for MC Psi(2S) \n";
  }
  else if(sample == 2 && channel == 3){
    tree.Add("/treeBd");
    cout << " Running for MC NoRes\n";

  }
  else {
    std::cout << " No valid file has been try" << std::endl;
    return;
  }


  ROOT::RDataFrame dDF(tree);
  //dDF.Describe().Print();
  auto nentries = dDF.Count();
  cout << " Total entries in Events " << *nentries <<endl;

  Double_t MpdgB = 5.27972;
  Double_t MpdgJ = 3.0969;
  Double_t Mpdgpsi = 3.6861;
  const double MpdgKstar = 0.89167;

  // auto dDF0 = dDF .Define("MpdgB",[&MpdgB] { return MpdgB; })
  //                 .Define("MpdgJ", [&MpdgJ] { return MpdgJ; })
  //                 .Define("Mpdgpsi", [&Mpdgpsi] { return Mpdgpsi; });
  
  auto dDF0 = dDF
    .Define("MpdgB",   [&MpdgB]   { return MpdgB; })
    .Define("MpdgJ",   [&MpdgJ]   { return MpdgJ; })
    .Define("Mpdgpsi", [&Mpdgpsi] { return Mpdgpsi; })
    .Define("MpdgKstar", [] { return 0.89167; }) // Define K* mass once
    .Define("masskstar_test", [MpdgKstar](double masskstar1, double masskstar2) {
        double diff1 = std::abs(masskstar1 - MpdgKstar);
        double diff2 = std::abs(masskstar2 - MpdgKstar);
        if(diff1 <= diff2) return masskstar1;
        else               return masskstar2;
    }, {"masskstar1", "masskstar2"})
    .Define("massB_test", [MpdgKstar](double masskstar1, double masskstar2, double massB1, double massB2) {
        double diff1 = std::abs(masskstar1 - MpdgKstar);
        double diff2 = std::abs(masskstar2 - MpdgKstar);
        if(diff1 <= diff2) return massB1;
        else               return massB2;
    }, {"masskstar1", "masskstar2", "massB1", "massB2"})
    // ... definiciones anteriores de masskstar_test ...

    // Definir CosThetaK Final basado en la mejor masa
    .Define("CosThetaK", [MpdgKstar](double masskstar1, double masskstar2, double ctk1, double ctk2) {
        double diff1 = std::abs(masskstar1 - MpdgKstar);
        double diff2 = std::abs(masskstar2 - MpdgKstar);
        // Si la hipótesis 1 está más cerca de la masa del K*, usamos el ángulo de la hipótesis 1
        if(diff1 <= diff2) return ctk1; 
        else               return ctk2;
    }, {"masskstar1", "masskstar2", "CosThetaK_H1", "CosThetaK_H2"})

    // Definir CosThetaL Final (aunque suele ser simétrico, es bueno ser consistente)
    .Define("CosThetaL", [MpdgKstar](double masskstar1, double masskstar2, double ctl1, double ctl2) {
        double diff1 = std::abs(masskstar1 - MpdgKstar);
        double diff2 = std::abs(masskstar2 - MpdgKstar);
        if(diff1 <= diff2) return ctl1;
        else               return ctl2;
    }, {"masskstar1", "masskstar2", "CosThetaL_H1", "CosThetaL_H2"})

    // Definir Phi Final
    .Define("Phi", [MpdgKstar](double masskstar1, double masskstar2, double phi1, double phi2) {
        double diff1 = std::abs(masskstar1 - MpdgKstar);
        double diff2 = std::abs(masskstar2 - MpdgKstar);
        if(diff1 <= diff2) return phi1;
        else               return phi2;
    }, {"masskstar1", "masskstar2", "Phi_H1", "Phi_H2"});


  auto selTrigger = "tri_DMu4_3_LM==1 && massJ>1.048808848";//sqrt(1.1)
  auto selVetoPhi = "(massvetokk < 1.010 || massvetokk > 1.030)";
  auto selJvetomass = " (massJ<2.8284 || massJ>3.3166) && (massJ<3.5355 || massJ>3.8729) "; //My window
  auto selvetokstar =  "masskstar_test >0.79555 && masskstar_test<0.99555";
  auto selvetoBmass = "massB_test >= 5.0 && massB_test <= 5.6";
  //auto selvetoBmass = "(massB1>5.150 && massB1<5.390) && (massB2>5.150 && massB2<5.390)";

  // //filtros Dr. Jhovanny
  // auto selVetoPhi = "(massvetokk > 1.035)";
  // auto selJvetomass = " (massJ>1.048808848) "; //My window
  // auto selvetokstar =  "masskstar_test >0.79555 && masskstar_test<0.99555";
  // auto selvetoBmass = "massB_test >= 5.0 && massB_test <= 5.6";
  // auto selNominal = "muon_dca<0.5 && Jprob>0.1 && (Bdl/BdlE)>5.0 && Bprob>0.1 && Pi1pt>1.2 && Pi2pt>1.2 ";



  //OPCIÓN A
  auto selJAntiRad1   = "(fabs((massB_test-MpdgB)-(massJ-MpdgJ)) > 0.17 ) || (massJ>MpdgJ) ";
  auto selJAntiRad2   = "(fabs((massB1-MpdgB)-(massJ-MpdgJ)) > 0.134 )  && (fabs((massB2-MpdgB)-(massJ-MpdgJ)) > 0.134 )   || (massJ>3.43) || (massJ<MpdgJ)";  
  auto selpsiAntiRad1 = "(fabs((massB_test-MpdgB)-(massJ-Mpdgpsi)) > 0.08) || (massJ>Mpdgpsi) ";  
  auto selpsiAntiRad2 = "(fabs((massB1-MpdgB)-(massJ-Mpdgpsi)) > 0.0941) && (fabs((massB2-MpdgB)-(massJ-Mpdgpsi)) > 0.0941)  || (massJ>3.92) || (massJ<Mpdgpsi)"; //Horacio's antiradiation veto
  auto selMCmatching  = "((dr_track1K < 0.04 && dr_track2Pi < 0.04) || (dr_track1Pi < 0.04 && dr_track2K < 0.04)) && drmu1 < 0.004 && drmu2 < 0.004";
  


  auto seldata = AddSelection(dDF0, 
                              selTrigger,     //filter1
                              selVetoPhi,     //filter2
                              selJvetomass,   //filter3
                              selJAntiRad1,   //filter4
                              selJAntiRad2,   //filter5
                              selpsiAntiRad1, //filter6
                              selpsiAntiRad2, //filter7
                              selvetokstar,   //filter8
                              selvetoBmass,   //filter9
                              //selNominal,     //filter10
                              selMCmatching,  
                              sample, 
                              version);
  
  
  //vector<string> columns = {"massB","massv0","B_kstar_mass","massJ","cosalfaB","cosalfaV0","Bdl","Bpt","v0pt","v0dl","Pi1pt","Pi2pt","Pi3pt","mu1pt","mu2pt","mu1eta","mu2eta","Jprob","v0prob","Bprob","priVtxCL", "muon_dca", "sigLxyBtmp", "DimuonMassErr", "mass2muK0s"};
  //vector<string> columns = {"masskstar_test","massB_test","massB1", "massB2","massvetopipi", "massvetokk",   "masskstar1", "masskstar2",   "massJ",   "cosalfaB",   "Bdl",   "Bpt",   "Pi1pt",   "Pi2pt",   "mu1pt",   "mu2pt",   "mu1eta",   "mu2eta",   "Jprob",   "Bprob",   "priVtxCL",   "muon_dca", "Frompv_Trk1","Frompv_Trk2" }; 
    vector<string> columns = {"masskstar_test","massB_test","massB1", "massB2","massvetopipi", "massvetokk",   "masskstar1", "masskstar2",   "massJ",   "cosalfaB",   "Bdl",   "Bpt",   "Pi1pt",   "Pi2pt",   "mu1pt",   "mu2pt",   "mu1eta",   "mu2eta",   "Jprob",   "Bprob",   "priVtxCL",   "muon_dca", "BdlE", "nVtx", "CosThetaK", "CosThetaL", "Phi"}; 

  TString ofile;

  if(sample==1){
    ofile = Form("AntiRadVeto_%s_%i_Era%i_v%i.root", sample_name.Data(), year, Era,version);
  }
  else{
    ofile = Form("AntiRadVeto_%s_%s_%i_Era%i_v%i.root", sample_name.Data(), channel_name.Data(), year, Era,version);
  }
  seldata.Snapshot("treeBd", ofile, columns);

  // TString ofile = Form("ntuple_mumuv0_ForXGboost_NOvetos_Year%1i_Sample%1i.root",year,sample);
}// End of main funcion






//************************************************************
// Other function implementations (see the definitions above)
//************************************************************

/* 
#DATA   
data_df_cut  = data_df[(data_df['massJ']<2.96) | (data_df['massJ']>3.21)]
data_df_cut  = data_df_cut[(data_df_cut['massJ']<3.59) | (data_df_cut['massJ']>3.78)]

antiRad_JPd  = (np.abs((data_df_cut.massB - 5.27934)-(data_df_cut.massJ-3.0969)) > 0.250) | (data_df_cut.massJ>3.0969)
antiRad_JPd2 = ((np.abs((data_df_cut.massB - 5.27934)-(data_df_cut.massJ-3.0969)) > 0.127) | (data_df_cut.massJ > 3.35)) | (data_df_cut.massJ<3.0969)
data_df_cut  = data_df_cut[(antiRad_JPd) & (antiRad_JPd2)]

antiRad_PPd  =  (np.abs((data_df_cut.massB - 5.27934)-(data_df_cut.massJ-3.6861)) > 0.085) | (data_df_cut.massJ>3.6861)
antiRad_PPd2 = ((np.abs((data_df_cut.massB - 5.27934)-(data_df_cut.massJ-3.6861)) > 0.054) | (data_df_cut.massJ > 3.92)) | (data_df_cut.massJ<3.6861)
data_df_cut = data_df_cut[(antiRad_PPd) & (antiRad_PPd2)]

//MC matching
yo hice los cortes de dr con muones y Ks0 asi
signal_df = mc_df[ (mc_df['dr_ks0']<0.01) & (mc_df['dr_m1']<0.01) & (mc_df['dr_m2']<0.01) ] (ojo son cortes durisimos, quizas los podemos relajar)
*/
