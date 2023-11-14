#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <iomanip>
#include <algorithm>

using namespace std;

int nrFeats;
int nrInstances = 0;


int id=0;

struct instance{
    int id;
    vector<bool> features;
    vector<int> attributes;
};

vector<vector<map<set<int>,int>>> cache;

vector<vector<instance>> instances;

int solve(int d,const vector<vector<instance>>& data, int ub);

ifstream in(R"(C:\Users\Horis\Documents\Algoritmus\Aslan\ODT\anneal.txt)");

void addInstance(const string& line){
    nrFeats = (line.length()-1)/2;
    istringstream ps(line);
    vector<bool> feats;
    vector<int> atts;
    int label;
    ps >> label;
    for(int i=0;i<nrFeats;i++){
        int feat;
        ps >> feat;
        if(feat) atts.push_back(i);
        feats.push_back(feat);
    }
    if(instances.size() <= label) instances.resize(label+1);
    instance i;
    i.attributes = atts;
    i.features = feats;
    i.id = id;
    id++;
    instances[label].push_back(i);
}

int optimal(const vector<vector<instance>>& data){
    int max = 0;
    int total = 0;
    for(const auto& label : data){
        total+=label.size();
        if(max < label.size()) max = label.size();
    }
    return total - max;
}

void splitData(int feat,const vector<vector<instance>>& data, vector<vector<instance>>& posData, vector<vector<instance>>& negData){
    for(int label=0; label < data.size(); label++){
        for(auto instance : data[label]){
            if(instance.features[feat]){
                if(posData.size() <= label) posData.resize(label+1);
                posData[label].push_back(instance);
            }
            else{
                if(negData.size() <= label) negData.resize(label+1);
                negData[label].push_back(instance);
            }
        }
    }
}

vector<vector<int>> CS;
vector<int> BR;
vector<int> BL;


struct prevDatasetFQ{
    vector<vector<instance>> prevDataSpec2;
    vector<vector<int>> FQpos;
    vector<vector<int>> FQneg;
    vector<int> FQneg1;
    vector<int> FQpos1;
};

pair<prevDatasetFQ,prevDatasetFQ> prevsSpec2;

set<int> extract(const vector<vector<instance>>& data){
    set<int> s;
    for(vector<instance> l : data){
        for(instance i : l){
            s.insert(i.id);
        }
    }
    return s;
}


void initialise(){
    prevsSpec2.first.prevDataSpec2.resize(2);
    prevsSpec2.second.prevDataSpec2.resize(2);
    prevsSpec2.first.FQpos.resize(2*nrFeats);
    prevsSpec2.first.FQneg.resize(2*nrFeats);
    prevsSpec2.first.FQpos1.resize(nrFeats);
    prevsSpec2.first.FQneg1.resize(nrFeats);
    prevsSpec2.first.FQneg1.assign(nrFeats, 0);
    prevsSpec2.first.FQpos1.assign(nrFeats, 0);
    prevsSpec2.second.FQpos.resize(2*nrFeats);
    prevsSpec2.second.FQneg.resize(2*nrFeats);
    prevsSpec2.second.FQpos1.resize(nrFeats);
    prevsSpec2.second.FQneg1.resize(nrFeats);
    prevsSpec2.second.FQneg1.assign(nrFeats, 0);
    prevsSpec2.second.FQpos1.assign(nrFeats, 0);
    CS.resize(2*nrFeats);
    for(int i=0;i<2*nrFeats;i++){
        prevsSpec2.first.FQpos[i].resize(2*nrFeats);
        prevsSpec2.first.FQneg[i].resize(2*nrFeats);
        prevsSpec2.first.FQpos[i].assign(2*nrFeats, 0);
        prevsSpec2.first.FQneg[i].assign(2*nrFeats, 0);
        prevsSpec2.second.FQpos[i].resize(2*nrFeats);
        prevsSpec2.second.FQneg[i].resize(2*nrFeats);
        prevsSpec2.second.FQpos[i].assign(2*nrFeats, 0);
        prevsSpec2.second.FQneg[i].assign(2*nrFeats, 0);
        CS[i].resize(2*nrFeats);
    }
}


void computeFQ(const vector<instance>& data, vector<vector<int>>& FQ, vector<int>& FQ1, bool incoming){
    for(const instance& ins : data){
        vector<int> v = ins.attributes;
        for(int i=0; i<v.size(); i++){
            if(incoming){
                FQ1[v[i]]++;
            }
            else{
                FQ1[v[i]]--;
            }
            for(int j=i+1; j<v.size(); j++){
                    if(incoming){
                        FQ[2*v[i]][2*v[j]]++;
                        FQ[2*v[j]][2*v[i]]++;
                    }
                    else{
                        FQ[2*v[i]][2*v[j]]--;
                        FQ[2*v[j]][2*v[i]]--;
                    }
            }

        }
    }
}

void computeFQcom(int D1, int D2,vector<vector<int>>& FQpos, vector<vector<int>>& FQneg,
                  vector<int>& FQneg1, vector<int>& FQpos1){
    for(int i=0; i<2*nrFeats; i++) {
        for (int j = 0; j < 2 * nrFeats; j++) {
            if(i==j) continue;
            if(i%2 == 1 && j%2 == 1) {
                FQpos[i][j] = D1 - FQpos1[i/2] - FQpos1[j / 2] + FQpos[i - 1][j - 1];
                FQneg[i][j] = D2 - FQneg1[i/2] - FQneg1[j / 2] + FQneg[i - 1][j - 1];
                FQpos[j][i] = FQpos[i][j];
                FQneg[j][i] = FQneg[i][j];

            }
            else if(i%2 == 1){
                FQpos[i][j] = FQpos1[j/2] -FQpos[i-1][j];
                FQneg[i][j] = FQneg1[j/2] -FQneg[i-1][j];
            }
            else if(j%2 == 1){
                FQpos[i][j] = FQpos1[i/2] -FQpos[i][j-1];
                FQneg[i][j] = FQneg1[i/2] -FQneg[i][j-1];
            }
        }
    }
}

void computeCS(vector<vector<int>>& FQpos, vector<vector<int>>& FQneg){
        for(int i=0; i<nrFeats; i++){
            for(int j=0; j<nrFeats;j++){
                if(i == j) continue;
                CS[2*i][2*j] = min(FQpos[2*i][2*j],FQneg[2*i][2*j]);
                CS[2*i+1][2*j] = min(FQpos[2*i+1][2*j],FQneg[2*i+1][2*j]);
                CS[2*i][2*j+1] = min(FQpos[2*i][2*j+1],FQneg[2*i][2*j+1]);
                CS[2*i+1][2*j+1] = min(FQpos[2*i+1][2*j+1],FQneg[2*i+1][2*j+1]);
                int MSleft = CS[2*i+1][2*j+1] + CS[2*i+1][2*j];
                int MSright = CS[2*i][2*j] + CS[2*i][2*j+1];
                if(MSleft < BL[i]) BL[i] = MSleft;
                if(MSright < BR[i]) BR[i] = MSright;
            }
        }
}


bool symDiff(vector<vector<instance>>& data, vector<vector<instance>>& inData, vector<vector<instance>>& outData) {
    int count1=0,count2=0;
    vector<vector<instance>> in1;
    vector<vector<instance>> in2;
    vector<vector<instance>> out1;
    vector<vector<instance>> out2;
    in1.resize(2);
    in2.resize(2);
    out1.resize(2);
    out2.resize(2);
    for(int l=0;l<2;l++){
       int i=0,j=0;
       vector<instance>& cur = data[l];
       vector<instance>& first = prevsSpec2.first.prevDataSpec2[l];
       vector<instance>& second = prevsSpec2.second.prevDataSpec2[l];
       while(i < cur.size() && j < first.size()){
           if(cur[i].id == first[j].id){
               count1++;
               i++;
               j++;
           }
           else if(cur[i].id < first[j].id){
               in1[l].push_back(cur[i]);
               i++;
           }
           else{
               out1[l].push_back(first[j]);
               j++;
           }
       }
       while(i < cur.size()){
           in1[l].push_back(cur[i]);
           i++;
       }
       while(j < first.size()){
           out1[l].push_back(first[j]);
           j++;
       }
       i=0,j=0;
        while(i < cur.size() && j < second.size()){
            if(cur[i].id == second[j].id){
                count2++;
                i++;
                j++;
            }
            else if(cur[i].id < second[j].id){
                in2[l].push_back(cur[i]);
                i++;
            }
            else{
                out2[l].push_back(second[j]);
                j++;
            }
        }
        while(i < cur.size()){
            in2[l].push_back(cur[i]);
            i++;
        }
        while(j < second.size()){
            out2[l].push_back(second[j]);
            j++;
        }
    }
    if(count1 > count2 ){
        for(int l=0; l<2; l++){
            for(const instance& i : in1[l]) inData[l].push_back(i);
            for(const instance& i : out1[l]) outData[l].push_back(i);
        }
//        inData = in1;
//        outData = out1;
        return true;
    }
    else{
        for(int l=0; l<2; l++){
            for(const instance& i : in2[l]) inData[l].push_back(i);
            for(const instance& i : out2[l]) outData[l].push_back(i);
        }
        /*inData = in2;
        outData = out2;*/
        return false;
    }
}

void copyData(vector<vector<instance>>& data,bool first){
    if(first){
        prevsSpec2.first.prevDataSpec2 = vector<vector<instance>>();
        prevsSpec2.first.prevDataSpec2.resize(2);
        for(int l =0 ; l < 2; l++){
            for(const instance& i : data[l]) prevsSpec2.first.prevDataSpec2[l].push_back(i);
        }
    }
    else{
        prevsSpec2.second.prevDataSpec2 = vector<vector<instance>>();
        prevsSpec2.second.prevDataSpec2.resize(2);
        for(int l =0 ; l < 2; l++){
            for(const instance& i : data[l]) prevsSpec2.second.prevDataSpec2[l].push_back(i);
        }
    }
}

void resetFQ(bool first, bool positive){
    prevDatasetFQ& p = first ? prevsSpec2.first : prevsSpec2.second;
    vector<vector<int>>& FQ = positive ? p.FQpos : p.FQneg;
    vector<int>& FQ1 = positive ? p.FQpos1 : p.FQneg1;
    FQ1.assign(nrFeats, 0);
    for(int i=0;i<2*nrFeats;i++){
       FQ[i].assign(2*nrFeats, 0);
    }
}

int special2(vector<vector<instance>> data){
    vector<instance>& posData = data[1];
    vector<instance>& negData = data[0];
    if(posData.empty() || negData.empty()) return 0;
    BR.assign(nrFeats,INT32_MAX/4);
    BL.assign(nrFeats,INT32_MAX/4);
    int sol = INT32_MAX-12;
    vector<vector<instance>> in;
    vector<vector<instance>> out;
    in.resize(2);
    out.resize(2);
    if(symDiff(data, in, out)){
        if(in[0].size() + out[0].size() < data[0].size()){
            computeFQ(in[0],prevsSpec2.first.FQneg,prevsSpec2.first.FQneg1,true);
            computeFQ(out[0],prevsSpec2.first.FQneg,prevsSpec2.first.FQneg1,false);
        }
        else{
            resetFQ(true,false);
            computeFQ(negData,prevsSpec2.first.FQneg,prevsSpec2.first.FQneg1,true);
        }
        if(in[1].size() + out[1].size() < data[1].size()){
            computeFQ(in[1],prevsSpec2.first.FQpos,prevsSpec2.first.FQpos1,true);
            computeFQ(out[1],prevsSpec2.first.FQpos,prevsSpec2.first.FQpos1,false);
        }
        else{
            resetFQ(true,true);
            computeFQ(posData,prevsSpec2.first.FQpos,prevsSpec2.first.FQpos1,true);
        }
        computeFQcom(posData.size(), negData.size(),prevsSpec2.first.FQpos
                     ,prevsSpec2.first.FQneg,prevsSpec2.first.FQneg1,prevsSpec2.first.FQpos1);
        computeCS(prevsSpec2.first.FQpos,prevsSpec2.first.FQneg);
        copyData(data,true);
    }
    else{
        if(in[0].size() + out[0].size() < data[0].size()){
            computeFQ(in[0],prevsSpec2.second.FQneg,prevsSpec2.second.FQneg1,true);
            computeFQ(out[0],prevsSpec2.second.FQneg,prevsSpec2.second.FQneg1,false);
        }
        else{
            resetFQ(false,false);
            computeFQ(negData,prevsSpec2.second.FQneg,prevsSpec2.second.FQneg1,true);
        }
        if(in[1].size() + out[1].size() < data[1].size()){
            computeFQ(in[1],prevsSpec2.second.FQpos,prevsSpec2.second.FQpos1,true);
            computeFQ(out[1],prevsSpec2.second.FQpos,prevsSpec2.second.FQpos1,false);
        }
        else{
            resetFQ(false,true);
            computeFQ(posData,prevsSpec2.second.FQpos,prevsSpec2.second.FQpos1,true);
        }
        computeFQcom(posData.size(), negData.size(),prevsSpec2.second.FQpos
                ,prevsSpec2.second.FQneg,prevsSpec2.second.FQneg1,prevsSpec2.second.FQpos1);
        computeCS(prevsSpec2.second.FQpos,prevsSpec2.second.FQneg);
        copyData(data,false);
    }


    for(int i=0; i<nrFeats;i++){
        if(BL[i] + BR[i] < sol) sol = BL[i] + BR[i];
    }
    return sol;
}

int solveFeat(int d,const vector<vector<instance>>& data, int ub, int i){
    vector<vector<instance>> posData;
    vector<vector<instance>> negData;
    splitData(i,data,posData,negData);
    int left,right;
    if(optimal(posData) > optimal(negData)){
        left = solve(d-1,posData,ub);
        if(left == -1) return -1;
        right = solve(d-1,negData,ub-left);
        if(right == -1) return -1;
    }
    else{
        right = solve(d-1,negData,ub);
        if(right == -1) return -1;
        left = solve(d-1,posData,ub-right);
        if(left == -1) return -1;
    }
//    if(left < 0 || right < 0) cout << "a";
//    if(d == 4) cout << ub << " " << left << " " << right << "\n";
    return left + right;
}

int solve(int d,const vector<vector<instance>>& data, int ub){
    if(ub < 0) return -1;
    if(data.empty() || data.size() == 1 || data.size() == 0) return 0;
    if(d==0){
        int leaf = optimal(data);
        return leaf <= ub ? leaf : -1;
    }
    set<int> presIns = extract(data);
    int dataSize = presIns.size();
    if(cache[d][dataSize].find(presIns)!=cache[d][dataSize].end()){
        int cached = cache[d][dataSize].find(presIns)->second;
        return cached <= ub ? cached : -1;
    }
    if(d==2 && data.size()==2){
        int spec2 = special2(data);
        return spec2 <= ub ? spec2 : -1;
    }
    int best = optimal(data);
    int ubr;
    for(int i=0;i<nrFeats;i++){
        ubr = min(ub,best-1);
        int t = solveFeat(d,data,ubr,i);
        //cout << t << "\n";
        if(t != -1) best = t;

    }
    if(best <= ub){
        cache[d][dataSize].insert({presIns, best});
    }
    else{
        best = -1;
    }
    //cout << d << " " << ubr << "\n";
    return best;
}

void initPrev(){
    for(int l=0; l<2; l++){
        for(const instance& i : instances[l]) prevsSpec2.first.prevDataSpec2[l].push_back(i);
        for(const instance& i : instances[l]) prevsSpec2.second.prevDataSpec2[l].push_back(i);
    }
    computeFQ(instances[1],prevsSpec2.first.FQpos,prevsSpec2.first.FQpos1,true);
    computeFQ(instances[0],prevsSpec2.first.FQneg,prevsSpec2.first.FQneg1,true);
    computeFQ(instances[1],prevsSpec2.second.FQpos,prevsSpec2.second.FQpos1,true);
    computeFQ(instances[0],prevsSpec2.second.FQneg,prevsSpec2.second.FQneg1,true);
}

void sizeCache(int d){
    cache.resize(d+1);
    for(auto& c : cache){
        c.resize(nrInstances+1);
    }
}

int main() {
    string line;
    time_t start, end;
    time(&start);
    while(getline(in,line)){
        addInstance(line);
        nrInstances++;
    }
    int d;
    cin >> d;
    sizeCache(d);
    initialise();
    initPrev();
    cout << solve(d,instances,nrInstances) << "\n";
    time(&end);
    auto time_taken = double(end - start);
    cout << "Time taken by program is : " << fixed
         << time_taken << setprecision(5);
    cout << " sec " << endl;
    return 0;
}
