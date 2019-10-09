//#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void read_data(string filename,
               int &weights_neurons,
               int &num_of_layers,
               vector<int> &neurons_in_layer,
               vector<float>&Input_,
               vector<int>&weights_neurons_vect,
               vector<vector<float> > &weights_per_neurons)
{
    fstream file;
    string word;
    file.open(filename.c_str());
    int num_of_neurons=0;



    file >> word;
    int val = std::stoi(word);

    num_of_layers=val;

    for(int i=0;i<num_of_layers;i++)
    {
       file >> word;
       val = std::stoi(word);
       neurons_in_layer.push_back(val);
       num_of_neurons=num_of_neurons+val;
    }

    for(int i=0;i<neurons_in_layer[0];i++)
    {
       file >> word;
       float val = std::stof(word);
       Input_.push_back(val);

    }
    weights_neurons=num_of_neurons-neurons_in_layer[num_of_layers-1];
    weights_per_neurons.resize(weights_neurons);
    for (int i = 0; i < weights_neurons; i++)
    {

        int col;
        int neurons=0;
        for(int k=0;k<num_of_layers;k++)
        {
            neurons+=neurons_in_layer[k];
            if(i<neurons)
            {
            col=neurons_in_layer[k+1];
            break;
            }

        }

        weights_neurons_vect.push_back(col);
        weights_per_neurons[i] = vector<float>(col);
        for (int j = 0; j < col; j++)
        {   file >> word;
                    float val = std::stof(word);
                    weights_per_neurons[i][j] =val ;
        }
    }



}

void split_weight_per_layer(const int & num_of_layers,
                            const vector<int> neurons_in_layer,
                            const vector<int>&weights_neurons_vect,
                            const vector<vector<float>> &weights_per_neurons,
                            vector<vector<vector<float>>>&weights_per_layer,
                            vector<vector<int>>&no_weights_per_neuron_per_layer
                            )
{


    for (int layer=0;layer<num_of_layers-1;layer++)
    {
      weights_per_layer[layer]=vector<vector<float>>(neurons_in_layer[layer]);
      no_weights_per_neuron_per_layer[layer]=vector<int>(neurons_in_layer[layer]);
      for(int row=0;row<neurons_in_layer[layer];row++)
      { int no_neurons=0;
        for(int i=0;i<layer;i++)
        {
         no_neurons+=neurons_in_layer[i];
        }
        no_weights_per_neuron_per_layer[layer][row]=weights_neurons_vect[no_neurons+row];
        weights_per_layer[layer][row]=vector<float>(weights_neurons_vect[no_neurons+row]);
        for(int col=0;col<weights_neurons_vect[no_neurons+row];col++)
        {
         weights_per_layer[layer][row][col]=weights_per_neurons[no_neurons+row][col];
        }
      }

    }


}

void feed_forward(const vector<vector<vector<float>>>&weights_per_layer,
                  const vector<float>&Input_,
                  vector<vector<float>>&output,
                  const int & num_of_layers
                   )
     {


    float sum=0.0;
    //

    for(int i=0;i<num_of_layers-1;i++){
      output[i]=vector<float>(weights_per_layer[i][0].size());
    }

    for(int layer=0;layer<(num_of_layers-1);layer++)
    {
      for(unsigned int i=0;i<weights_per_layer[layer][0].size();i++)
      {
        for(unsigned int j=0;j<weights_per_layer[layer].size();j++)
         {
          if(layer==0)
           {
            sum+=((float)Input_[j])*weights_per_layer[0][j][i];
           }
          else{
            sum+=output[layer-1][j]*weights_per_layer[1][j][i];
              }
          }
      output[layer][i]=sum;
      sum=0.0;
      //cout<<output0[i]<<endl;
    }
    }



     }

int main()
{

    string filename;
    filename = "/mnt/407242D87242D1F8/study/C++/C++_Task/net3.txt";


    int weights_neurons;
    int num_of_layers;
    vector<int> neurons_in_layer;
    vector<float>Input_;
    vector<int>weights_neurons_vect;
    vector<vector<float> > weights_per_neurons;
    read_data(filename,weights_neurons,num_of_layers,neurons_in_layer,Input_,weights_neurons_vect,weights_per_neurons);

    vector<vector<vector<float>>>weights_per_layer(num_of_layers-1);
    vector<vector<int>>no_weights_per_neuron_per_layer(num_of_layers-1);
    vector<vector<float>>output(num_of_layers-1);
    split_weight_per_layer( num_of_layers,
                            neurons_in_layer,
                            weights_neurons_vect,
                            weights_per_neurons,
                            weights_per_layer,
                            no_weights_per_neuron_per_layer
                            );
   feed_forward(weights_per_layer,
                Input_,
                output,
                num_of_layers
                   );


    for(int i=0;i<3;i++)
      cout<<output[1][i]<<endl;


    return 0;
}
