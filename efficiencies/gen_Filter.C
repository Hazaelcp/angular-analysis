#include <iostream>
#include <string>
#include <vector>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RVec.hxx>
#include "TFile.h"

using namespace std;
using namespace ROOT;

void gen_Filter() {

    string inputPath  = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/angular/efficiencies/datasets/GenLevel_Angular_Merged.root";
    string outputPath = "/home/ghcp/Documentos/CINVESTAV/ANALISYS_B0tomumuKstar/angular/efficiencies/datasets/GenLevel_Angular_Merged_Filtered.root";
    string treeName = "ntuple"; 
    ROOT::EnableImplicitMT(8);

    ROOT::RDataFrame dDF(treeName, inputPath);
    string accCuts = "gen_muon1_p4.Pt() > 3.5 && gen_muon2_p4.Pt() > 3.5 && "
                     "abs(gen_muon1_p4.Eta()) <= 2.5 && abs(gen_muon2_p4.Eta()) <= 2.5 && "
                     "gen_kstar_p4.Pt() > 0.5 && abs(gen_kstar_p4.Eta()) <= 2.5";

    auto dDF_filtered = dDF.Filter(accCuts, "Cortes_Aceptacion");
    auto report = dDF_filtered.Report();
    report->Print();
        
    ROOT::RDF::RSnapshotOptions opts;
    opts.fMode = "RECREATE"; 
    
    dDF_filtered.Snapshot(treeName, outputPath, "", opts);

    std::cout << ">>> Proceso terminado." << std::endl;
}